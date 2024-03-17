# Import ve Dosya Okuma

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from pandas import Timestamp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("dataset/MTA_15.csv")

df = df_.copy()
df.head()
df.info()
df.columns

# Veri Seti Hikayesi #
# WomenTechWomenYes (WTWY) derneği, her yaz düzenlediği yıllık gala etkinliğiyle kadınların teknolojiye katılımını artırmayı ve farkındalık yaratmayı hedeflemektedir.
# Gala için bağış toplama çabalarının önemli bir parçası olan sokak ekipleri, metro istasyonlarında konumlanarak katılımcıların e-posta adreslerini toplamakta ve galaya ücretsiz bilet sunmaktadır.
# WTWY, sokak ekiplerinin yerleşimini optimize etmek için MTA metro verilerinden faydalanmaya karar vermiştir.
# Bu veri seti, galaya katılabilecek ve amaca katkı sağlayabilecek en fazla kişiye ulaşmayı amaçlamaktadır.

# Değişkenler

# transit_timestamp : Ödemenin gerçekleştiği tarih ve saat yerel saat dilimine göre belirtilmiştir. | object
# transit_mode : Metro, Staten Island Railway ve Roosevelt Island Tramvayı arasındaki farkları açıklar. | object
# station_complex_id : İstasyon komplexleri için unique tanımlayıcı | object
# station_complex : Giriş yapma işleminin gerçekleştiği metro kompleksi. Times Square ve Fulton Center gibi büyük metro kompleksleri, birden fazla metro hattını içerebilir. Metro kompleksi adı, komplekste duran hatları parantez içinde belirtir, örneğin Zerega Av (6). | object
# borough : ilçe | object
# payment_method : Giriş için kullanılan ödeme yönteminin OMNY veya MetroCard'dan olup olmadığını belirtir. | object
# fare_class_category : Seyahat için kullanılan ücret ödeme sınıfı. Birleştirilmiş kategoriler şunlardır: | object
#  • MetroCard – Fair Fare;
#  • MetroCard – Full Fare;
#  • MetroCard – Other;
#  • MetroCard – Senior & Disability;
#  • MetroCard – Students;
#  • MetroCard – Unlimited 30-Day;
#  • MetroCard – Unlimited 7-Day;
#  • OMNY – Full Fare; • OMNY – Other;
#  • OMNY – Seniors & Disabilities
# Bu kategorilerin birleştirilmesinin amacı, belirli bir özellikteki veya demografik özelliğe sahip kullanıcıların seyahat ücretleriyle ilgili bilgileri daha kolay analiz etmeyi sağlamaktır.
# ridership : yolcu sayısı | int64
# transfers : int64 | Bu değişken, bir metro kompleksine ücretsiz olarak otobüsten metroya veya ücretsiz ağ dışı bir transfer yoluyla giren kişi sayısını temsil ediyor. Toplam yolcu sayısının bir alt kümesini oluşturuyor; çünkü bu transferler, zaten bir önceki yolcu sayısı sütununa dahil edilmiş durumda.
# Bu veri, metro istasyonlarında gerçekleşen bu transferleri izlemek ve analiz etmek için kullanılabilir.
# latitude : enlem | float64
# longitude : boylam | float64
# Georeference : coğrafi kodlama bilgileri | object

# Hedef Kitlesi : 01 Şubat 2024 | 14 Şubat 2024 arasındaki verileri kapsamaktadır.

# Adım 1 : Genel resmi inceleme

def check_df(dataframe, head=5):
    print("-" * 25 + "Info" + "-" * 25)
    print(dataframe.info())
    print("-" * 25 + "Shape" + "-" * 25)
    print(dataframe.shape)
    print("-" * 25 + "The First data" + "-" * 25)
    print(dataframe.head(head))
    print("-" * 25 + "The Last data" + "-" * 25)
    print(dataframe.tail(head))
    print("-" * 25 + "Missing values" + "-" * 25)
    print(dataframe.isnull().sum())
    print("-" * 25 + "Describe the data" + "-" * 25)
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("-" * 25 + "Distinct Values" + "-" * 25)
    print(dataframe.nunique())

check_df(df)


# Adım 2: Tarih düzenlemeleri

df['transit_timestamp'] = pd.to_datetime(df['transit_timestamp'])

min_date = df['transit_timestamp'].min()
max_date = df['transit_timestamp'].max()

df["date"] = df["transit_timestamp"].dt.date
df["day_of_week"] = df["transit_timestamp"].dt.day_name()

df["day_of_week"].value_counts()

# Adım 3: Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi

def grab_col_names(dataframe, cat_th=11, car_th=30):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframedir.
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int,float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    car_but_car : list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    --------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içinde
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtype in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("###################")

for col in cat_cols:
    cat_summary(df, col)


# Kategorileri Görselleştirme
def visualize_categorical(dataframe, col_name):
    """
    Verilen kategorik değişkenin dağılımını görselleştirir.
    Parameters:
    dataframe : pandas.DataFrame
        Veri çerçevesi
    col_name : str
        Görselleştirilecek kategorik değişkenin adı
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col_name, data=dataframe)
    plt.title(f'Distribution of {col_name}')
    plt.xticks(rotation=45)
    plt.show()

# Kategorik değişkenleri görselleştirme
for col in cat_cols:
    visualize_categorical(df, col)


# Adım 4 : Aykırı Değer ve Eksik Değer Gözlemi

# Aykırı Değer Gözlem Analizi
def outlier_graph(df):
    #cat_cols, num_cols, cat_but_car = grab_col_names(df)

    plt.figure(figsize=(15, 20))
    for i, column in enumerate(num_cols):
        plt.subplot(len(num_cols) // 2 + 1, 2, i+1)
        sns.boxplot(x=df[column], color="red")
        plt.title(f"Boxplot of {column}", pad=20)

    plt.tight_layout()

outlier_graph(df)
plt.show()

# Eksik Değer Gözlem Analizi

msno.bar(df)
plt.show()

# Veri setimizde eksik değer bulunmamaktadır.

# Adım 5: Çeşitli Analizlere Göre Görselleştirme

# a: En yoğun ilk 5 istasyon komplekslerine göre toplam yolcu hesaplama ve görselleştirme

#hepsi
station_ridership = df.groupby('station_complex')['ridership'].sum().sort_values(ascending=False)

#ilk 5 data
station_ridership_5data = df.groupby('station_complex')['ridership'].sum().sort_values(ascending=False).head(5)

station_complexes = station_ridership_5data.index
ridership_values = station_ridership_5data.values

plt.figure(figsize=(10, 6))
bars = plt.bar(station_complexes, ridership_values, color='purple')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')

plt.xlabel('Station Complexes')
plt.ylabel('Total Ridership')
plt.title('Total Passengers by Top 5 Intensive Station Complexes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()


# Daha Ayrıntılı Görselleştirme

station_daily_passengers = df.groupby(['date', 'station_complex'])['ridership'].sum().reset_index()

# En yoğun istasyonları bulma ve verileri seçme
top_stations = station_daily_passengers.groupby('station_complex')['ridership'].sum().nlargest(5).index

top_stations_data = station_daily_passengers[station_daily_passengers['station_complex'].isin(top_stations)]

# İstasyon bazında günlük yolcu sayısı çubuk grafik görselleştirme

plt.figure(figsize=(12, 6))
sns.barplot(x='date', y='ridership', hue='station_complex', data=top_stations_data, palette='muted')
plt.title('Top Stations - Daily Passenger Count')
plt.xlabel('Date')
plt.ylabel('Total Passenger Count')
plt.xticks(rotation=45)
plt.legend(title='Station Complex', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom')

plt.tight_layout()
plt.show()

# b : Günlere göre analizler

# Günlere göre toplam yolcu sayısı hesaplama
total_passengers_per_day = df.groupby('date')['ridership'].sum()

total_passengers_per_day.sort_values(ascending=False)
# Toplam yolcu sayısı çubuk grafik görselleştirme

plt.figure(figsize=(12, 6))
bars = total_passengers_per_day.plot(kind='bar', color='skyblue')
plt.title('Total Passenger Count - Per Day')
plt.xlabel('Date')
plt.ylabel('Total Passenger Count')
plt.xticks(rotation=40)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom')

plt.tight_layout()
plt.show()

#plt.savefig('/Users/melisacevik/desktop/MTA-visualizations/total_passengers_per_day.png')

# Hafta içi ve hafta sonu yolcu sayılarını karşılaştırma
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

passengers_weekend_vs_weekday = df.groupby('is_weekend')["ridership"].sum()

# Hafta içi ve hafta sonu yolcu sayısı çubuk grafik görselleştirme

plt.figure(figsize=(8, 6))
passengers_weekend_vs_weekday.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Weekend vs Weekday Comparison - Total Passenger Count')
plt.xlabel('Weekend (1) / Weekday (0)')
plt.ylabel('Total Passenger Count')
plt.xticks(ticks=[0, 1], labels=['Weekday', 'Weekend'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom')

plt.tight_layout()
plt.show()



