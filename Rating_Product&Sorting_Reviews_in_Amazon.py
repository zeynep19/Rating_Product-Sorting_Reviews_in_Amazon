########################################################################################
# Rating Product & Sorting Reviews in Amazon
########################################################################################
# Business Problem: Trying to calculate product ratings more accurately and ordering product reviews more accurately.
########################################################################################
# Dataset Story: This dataset, which includes Amazon product data, includes product categories and various metadata.
# The product with the most reviews in the electronics category has user ratings and reviews.
########################################################################################
# Variables
# reviewerID – User ID Ex: A2SUAM1J3GNN3B
# asin – Product ID. Ex: 0000013714
# reviewerName – Username
# helpful – Useful rating degree Ex: 2/3
# reviewText – Review User-written review text
# overall – Product rating
# summary – Evaluation summary
# unixReviewTime – Evaluation time Unix time
# reviewTime – Review time Raw
# day_diff – Number of days since evaluation
# helpful_yes – The number of times the review was found helpful
# total_vote – Number of votes given to the review
########################################################################################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv("pythonProject/datasets/amazon_review.csv")
df.head()

######################################################################################################
# Görev-1 Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
######################################################################################################

df["asin"].value_counts()
# The productID with the most reviews in the electronics category: B007WTAJTO

df["overall"].value_counts()
# Ratings
# 5.00000    3922
# 4.00000     527
# 1.00000     244
# 3.00000     142
# 2.00000      80

df["overall"].mean()
# Average Rating : 4.587589013224822

# Time-Based Weighted Average
# day_diff: Number of days since assessment
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)
# Time-Based Weighted Average Rating : 4.6987161061560725

# Yorum: Zaman tabanlı ağırlıklandırılmış ortalama oylama sonucunun genel ortalama oylamaya göre daha iyi
# çıkması ürünün önceye göre popülerliğinin arttığı anlamına gelebilir veya üründe iyileştirmeler yapılmış ise
# zamana göre memnuniyetin artması beklenir bir durumdur.

######################################################################################################
# Görev-2 Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
######################################################################################################

# Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], (x["total_vote"] - x["helpful_yes"])), axis=1)

df[["reviewerName", "helpful", "reviewText", "overall", "summary", "reviewTime", "wilson_lower_bound"]].\
    sort_values("wilson_lower_bound", ascending=False).head(20)










