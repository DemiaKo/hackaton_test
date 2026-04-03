import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import plotly.express as px
import pymap3d as pm
from pymavlink import mavutil
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from scipy.signal import medfilt


# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0)**2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ==========================================
# ІНТЕРФЕЙС STREAMLIT
# ==========================================

st.set_page_config(
    page_title="BEST Hackathon: UAV Telemetry",
    layout="wide",
    page_icon="🚁"
)

st.title("🚁 Система аналізу телеметрії та 3D-візуалізації польотів БПЛА")
st.markdown(
    "Завантажте сирий `.BIN` лог-файл контролера Ardupilot для автоматичного "
    "розбору, розрахунку метрик та 3D-візуалізації траєкторії."
)

uploaded_file = st.file_uploader(
    "Оберіть файл (наприклад, 00000001.BIN)", type=['bin', 'BIN']
)

if uploaded_file is not None:
    with st.spinner("Аналізуємо чорну скриньку... Це може зайняти кілька секунд."):

        # -----------------------------------------------
        # 1. ТИМЧАСОВИЙ ФАЙЛ (потрібен для pymavlink)
        # -----------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix='.BIN') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # -----------------------------------------------
        # 2. ПАРСИНГ ЛОГІВ
        # -----------------------------------------------
        mlog = mavutil.mavlink_connection(tmp_file_path)
        gps_data, imu_data, att_data, baro_data = [], [], [], []

        while True:
            msg = mlog.recv_match(
                type=['GPS', 'IMU', 'ATT', 'BARO'], blocking=False
            )
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type == 'GPS':
                gps_data.append(msg.to_dict())
            elif msg_type == 'IMU':
                imu_data.append(msg.to_dict())
            elif msg_type == 'ATT':
                att_data.append(msg.to_dict())
            elif msg_type == 'BARO':
                baro_data.append(msg.to_dict())

        # -----------------------------------------------
        # 3. ОБРОБКА GPS
        # -----------------------------------------------
        # Фільтруємо по якості фіксу: 3=3D Fix, 4=DGPS, 5=RTK
        df_gps_raw = pd.DataFrame(gps_data)
        df_gps_raw['TimeS'] = df_gps_raw['TimeUS'] / 1_000_000
        if 'Status' in df_gps_raw.columns:
            df_gps_raw = df_gps_raw[df_gps_raw['Status'] >= 3].copy()
        df_gps_raw = df_gps_raw.reset_index(drop=True)

        # Haversine для загальної дистанції
        df_gps = df_gps_raw[["TimeS", "Lat", "Lng", "Alt", "Spd", "VZ"]].copy()
        df_gps['Lat_next'] = df_gps['Lat'].shift(-1)
        df_gps['Lng_next'] = df_gps['Lng'].shift(-1)
        df_gps['Step_Distance'] = calculate_haversine_distance(
            df_gps['Lat'], df_gps['Lng'],
            df_gps['Lat_next'], df_gps['Lng_next']
        )
        total_distance = df_gps['Step_Distance'].fillna(0).sum()
        flight_duration = df_gps['TimeS'].max() - df_gps['TimeS'].min()

        # Горизонтальна швидкість — поле Spd від GPS-модуля (доплер)
        # 99й перцентиль відкидає поодинокі спайки
        max_horizontal_speed = df_gps_raw['Spd'].quantile(0.99)

        # Вертикальна швидкість — поле VZ від GPS
        # VZ в ArduPilot може бути у см/с — визначаємо автоматично
        vz_99 = df_gps_raw['VZ'].abs().quantile(0.99)
        max_vertical_speed = vz_99 / 100.0 if vz_99 > 100 else vz_99

        # -----------------------------------------------
        # 4. ОБРОБКА BARO — висота відносно точки старту
        # -----------------------------------------------
        # Барометр точніший за GPS по висоті (похибка < 1м vs 10–100м у GPS)
        df_baro = pd.DataFrame(baro_data)
        df_baro['TimeS'] = df_baro['TimeUS'] / 1_000_000
        df_baro = (df_baro[["TimeS", "Alt"]]
                   .rename(columns={"Alt": "AltBaro"})
                   .sort_values('TimeS')
                   .reset_index(drop=True))

        # Медіанний фільтр прибирає аномальні відліки
        df_baro['AltBaro_smooth'] = medfilt(
            df_baro['AltBaro'].values, kernel_size=11
        )

        # Відкидаємо фізично неможливі стрибки (> 50 м/с)
        df_baro['VelZ_baro'] = (df_baro['AltBaro_smooth'].diff()
                                / df_baro['TimeS'].diff())
        valid_baro = df_baro[df_baro['VelZ_baro'].abs() < 50]

        # Обрізаємо BARO по часовому вікну GPS (без прогріву на землі)
        gps_start = df_gps['TimeS'].min()
        gps_end   = df_gps['TimeS'].max()
        valid_baro = valid_baro[
            (valid_baro['TimeS'] >= gps_start) &
            (valid_baro['TimeS'] <= gps_end)
        ]

        # Висота відносно старту (не max-min, бо BARO — абсолютна)
        baro_start_alt = valid_baro['AltBaro_smooth'].head(10).mean()
        max_altitude_gain = valid_baro['AltBaro_smooth'].max() - baro_start_alt

        # -----------------------------------------------
        # 5. ОБРОБКА IMU + ATT — Sensor Fusion
        # -----------------------------------------------
        # IMU дає прискорення в Body Frame.
        # ATT дає кути. Об'єднуємо → переходимо в Earth Frame → прибираємо gravity.
        #
        # ЧОМУ НЕ ІНТЕГРУЄМО ДЛЯ ШВИДКОСТІ:
        # Похибка 0.01 м/с² × 60с = ~36 м/с дрейфу. Тому швидкості — з GPS.
        # IMU тут використовується лише для прискорення.

        df_imu = pd.DataFrame(imu_data)
        df_imu['TimeS'] = df_imu['TimeUS'] / 1_000_000
        df_imu = df_imu[["TimeS", "AccX", "AccY", "AccZ"]]

        df_att = pd.DataFrame(att_data)
        df_att['TimeS'] = df_att['TimeUS'] / 1_000_000
        df_att = df_att[["TimeS", "Roll", "Pitch", "Yaw"]]

        df_imu = df_imu.sort_values('TimeS')
        df_att = df_att.sort_values('TimeS')

        df_merged = pd.merge_asof(
            df_imu, df_att, on='TimeS', direction='nearest'
        )

        # Конвенція ZYX (Yaw→Pitch→Roll) відповідає стандарту ArduPilot
        angles = np.array(df_merged[['Yaw', 'Pitch', 'Roll']], dtype=float)
        rotations = Rotation.from_euler('ZYX', angles, degrees=True)
        accels_body = np.array(
            df_merged[['AccX', 'AccY', 'AccZ']], dtype=float
        )
        accels_earth = rotations.apply(accels_body)

        df_merged['AccX_earth'] = accels_earth[:, 0]
        df_merged['AccY_earth'] = accels_earth[:, 1]
        df_merged['AccZ_earth'] = accels_earth[:, 2]

        # Перші 50 семплів = дрон нерухомий → bias містить gravity
        bias_x = df_merged['AccX_earth'].head(50).mean()
        bias_y = df_merged['AccY_earth'].head(50).mean()
        bias_z = df_merged['AccZ_earth'].head(50).mean()  # ≈ -9.81 м/с²

        df_merged['AccX_clean'] = df_merged['AccX_earth'] - bias_x
        df_merged['AccY_clean'] = df_merged['AccY_earth'] - bias_y
        df_merged['AccZ_clean'] = df_merged['AccZ_earth'] - bias_z

        # Magnitude прискорення — миттєве і згладжене (вікно 5)
        df_merged['Acc_magnitude'] = np.sqrt(
            df_merged['AccX_clean']**2 +
            df_merged['AccY_clean']**2 +
            df_merged['AccZ_clean']**2
        )
        max_acceleration = (df_merged['Acc_magnitude']
                            .rolling(window=5, center=True)
                            .mean()
                            .max())

        # -----------------------------------------------
        # 6. МЕТРИКИ — вивід
        # -----------------------------------------------
        st.subheader("📊 Підсумкові кінематичні показники")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Пройдена дистанція",    f"{total_distance:.1f} м")
        col2.metric("Час польоту",           f"{flight_duration:.1f} с")
        col3.metric("Макс. набір висоти",    f"{max_altitude_gain:.1f} м")
        col4.metric("Макс. прискорення",     f"{max_acceleration:.2f} м/с²")

        col5, col6 = st.columns(2)
        col5.metric("Макс. горизонтальна швидкість (GPS Spd)",
                    f"{max_horizontal_speed:.2f} м/с")
        col6.metric("Макс. вертикальна швидкість (GPS VZ)",
                    f"{max_vertical_speed:.2f} м/с")

        st.divider()

        # -----------------------------------------------
        # 7. 3D ВІЗУАЛІЗАЦІЯ — конвертація WGS-84 → ENU
        # -----------------------------------------------
        # ENU (East-North-Up) — локальна декартова система від точки старту.
        # geodetic2enu перетворює глобальні координати у метри відносно (lat0, lon0, alt0).
        st.subheader("🗺️ 3D Траєкторія польоту (ENU)")

        lat0 = df_gps['Lat'].iloc[0]
        lon0 = df_gps['Lng'].iloc[0]
        alt0 = df_gps['Alt'].iloc[0]

        east, north, up = pm.geodetic2enu(
            df_gps['Lat'], df_gps['Lng'], df_gps['Alt'],
            lat0, lon0, alt0
        )

        df_gps['East']  = east
        df_gps['North'] = north
        df_gps['Up']    = up

        df_gps_plot = df_gps.dropna(subset=['East', 'North', 'Up', 'Spd']).copy()
        df_gps_plot['Spd'] = pd.to_numeric(
            df_gps_plot['Spd'], errors='coerce'
        ).fillna(0)

        fig = px.scatter_3d(
            df_gps_plot,
            x='East', y='North', z='Up',
            color='Spd',
            title='Просторова траєкторія відносно точки зльоту (0, 0, 0)',
            labels={
                'East':  'Схід (м)',
                'North': 'Північ (м)',
                'Up':    'Висота (м)',
                'Spd':   'Швидкість (м/с)'
            },
            color_continuous_scale='turbo'
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------
        # 8. ТЕОРЕТИЧНЕ ОБҐРУНТУВАННЯ
        # -----------------------------------------------
        with st.expander("Розгорнути теоретичне обґрунтування (для журі)"):
            st.markdown("""
### Теоретичне обґрунтування архітектури

**1. Чому IMU не використовується для швидкості:**
Сирі дані акселерометра містять шум та гравітаційну складову (~9.81 м/с²).
Навіть після видалення bias, похибка 0.01 м/с² за 60 секунд дає ~36 м/с
дрейфу (*INS drift*). Без зворотного зв'язку (як у EKF) результат непридатний.
Тому швидкості беруться напряму з GPS-полів `Spd` і `VZ`.

**2. Sensor Fusion (Body Frame → Earth Frame):**
За допомогою `scipy.spatial.transform.Rotation` вектори прискорення IMU
переводяться з локальної системи дрона (*Body Frame*) у глобальну (*Earth Frame*)
з використанням кутів Roll/Pitch/Yaw (повідомлення `ATT`, конвенція ZYX).
Лише після вирівнювання осей віднімається гравітаційний вектор.

**3. Чому BARO, а не GPS для висоти:**
`GPS.Alt` — абсолютна висота над рівнем моря з похибкою 10–100 м.
`BARO.Alt` — барометрична висота з похибкою < 1 м, ідеальна для відносного
набору висоти. Додатково застосовується медіанний фільтр (kernel=11) для
прибирання аномальних відліків.

**4. Чому перцентиль, а не максимум для швидкостей:**
GPS може давати поодинокі хибні відліки (спайки). 99й перцентиль дає
статистично коректний максимум без довільного обрізання (*clip*).

**5. Конвертація WGS-84 → ENU:**
`pymap3d.geodetic2enu` перетворює глобальні координати (градуси) у метри
відносно точки старту. Вісь X = Схід, Y = Північ, Z = Вгору.
            """)