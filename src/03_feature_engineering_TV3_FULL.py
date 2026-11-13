# 0. Setup
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DATA_FILE = ROOT / "dataset.csv"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

print("Dataset:", DATA_FILE)
print("Output :", OUT_DIR)


# 1.1. Đọc CSV với encoding phù hợp
try:
    raw = pd.read_csv(DATA_FILE)
except UnicodeDecodeError:
    raw = pd.read_csv(DATA_FILE, encoding="latin1")

raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
df = raw.copy()

df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["sales"] = df["quantity"] * df["price"]

print("Shape:", df.shape)
display(df.head())


# 1.2. Data quality theo TV2
quality = pd.DataFrame({
    "column": df.columns,
    "missing": [df[c].isna().sum() for c in df.columns],
    "missing_%": [round(df[c].isna().mean()*100, 2) for c in df.columns],
    "unique": [df[c].nunique(dropna=True) for c in df.columns],
    "dtype": [str(df[c].dtype) for c in df.columns]
})
display(quality)

print("Duplicates:", df.duplicated().sum())
print("Negative quantity:", (df["quantity"] < 0).sum())
print("Cancelled invoices:", df["invoice"].astype(str).str.upper().str.startswith("C").sum())
print("Date:", df["invoicedate"].min(), "->", df["invoicedate"].max())


# 2.1. Phân phối sales
print(df["sales"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

plt.figure(figsize=(10,4))
plt.hist(df["sales"].clip(df["sales"].quantile(.01), df["sales"].quantile(.99)), bins=60)
plt.title("Phân phối Sales sau khi clip để quan sát EDA")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 2.2. Sales theo tháng
tmp = df.dropna(subset=["invoicedate"]).copy()
tmp["month"] = tmp["invoicedate"].dt.to_period("M").astype(str)
monthly = tmp.groupby("month")["sales"].sum()

plt.figure(figsize=(12,4))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Total Sales by Month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3.1. Apply TV2-compatible cleaning
before = len(df)
df = df.dropna(subset=["invoicedate", "quantity", "price", "sales"]).drop_duplicates().copy()

df["is_cancelled"] = df["invoice"].astype(str).str.upper().str.startswith("C").astype(int)
df["is_return"] = ((df["quantity"] < 0) | (df["is_cancelled"] == 1)).astype(int)

print("Rows before:", before)
print("Rows after :", len(df))
print("Dropped    :", before-len(df))


# 4.1. Tạo tuần
df["week_start"] = df["invoicedate"].dt.to_period("W-SUN").dt.start_time

weekly = df.groupby(["stockcode","week_start"]).agg(
    sales=("sales","sum"),
    quantity=("quantity","sum"),
    transaction_count=("invoice","nunique"),
    active_days=("invoicedate", lambda s: s.dt.date.nunique()),
    avg_price=("price","mean"),
    unique_customers=("customer_id","nunique"),
    return_transaction_count=("is_return","sum"),
    cancel_transaction_count=("is_cancelled","sum")
).reset_index()

desc = df.groupby("stockcode")["description"].agg(
    lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else ""
)
weekly = weekly.merge(desc.rename("description"), on="stockcode", how="left")
weekly["description_length"] = weekly["description"].fillna("").str.len()

weekly = weekly.sort_values(["stockcode","week_start"]).reset_index(drop=True)
print("Product–Week:", weekly.shape)
display(weekly.head())


# 5.1. Time features
weekly["year"] = weekly["week_start"].dt.year
weekly["month"] = weekly["week_start"].dt.month
weekly["quarter"] = weekly["week_start"].dt.quarter
weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
weekly["is_year_end"] = weekly["month"].isin([11,12]).astype(int)
weekly["retail_peak_season_proxy"] = weekly["month"].isin([11,12]).astype(int)

weekly["trend_index"] = (
    (weekly["week_start"] - weekly["week_start"].min()).dt.days // 7
).astype(int)

display(weekly[["week_start","year","month","quarter","week_of_year",
                "is_year_end","retail_peak_season_proxy","trend_index"]].head())


# 6.1. Product historical features
g = weekly.groupby("stockcode", group_keys=False)

weekly["avg_price_lag_1"] = g["avg_price"].shift(1)
weekly["transaction_count_lag_1"] = g["transaction_count"].shift(1)
weekly["unique_customers_lag_1"] = g["unique_customers"].shift(1)

weekly["product_avg_price_hist"] = g["avg_price"].transform(
    lambda s: s.shift(1).expanding(min_periods=1).mean()
)
weekly["product_active_weeks_hist"] = g.cumcount()

weekly["price_change_vs_hist"] = (
    weekly["avg_price_lag_1"] / weekly["product_avg_price_hist"].replace(0,np.nan) - 1
)

display(weekly[[
    "stockcode","week_start","avg_price_lag_1",
    "product_avg_price_hist","transaction_count_lag_1",
    "product_active_weeks_hist"
]].head(10))


# 7.1. Customer history tại transaction level
tx = df.sort_values(["customer_id","invoicedate"]).copy()

tx["prior_sales"] = (
    tx.groupby("customer_id")["sales"].cumsum() - tx["sales"]
)
tx["prior_tx_count"] = tx.groupby("customer_id").cumcount()
tx["prev_date"] = tx.groupby("customer_id")["invoicedate"].shift(1)
tx["prior_recency_days"] = (
    (tx["invoicedate"] - tx["prev_date"]).dt.total_seconds() / 86400
)

rfm_prod = tx.groupby(["stockcode","week_start"]).agg(
    avg_customer_prior_recency_days=("prior_recency_days","mean"),
    avg_customer_prior_frequency=("prior_tx_count","mean"),
    avg_customer_prior_monetary=("prior_sales","mean")
).reset_index()

weekly = weekly.merge(rfm_prod, on=["stockcode","week_start"], how="left")

def qscore(s, reverse=False):
    valid = s.dropna()
    if valid.nunique() < 2:
        return pd.Series(3, index=s.index, dtype=float)
    ranks = valid.rank(method="average", pct=True)
    score = np.ceil(ranks*5).astype(int)
    if reverse:
        score = 6-score
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[valid.index] = score
    return out

weekly["R_score"] = qscore(weekly["avg_customer_prior_recency_days"], reverse=True)
weekly["F_score"] = qscore(weekly["avg_customer_prior_frequency"])
weekly["M_score"] = qscore(weekly["avg_customer_prior_monetary"])
weekly["RFM_score"] = weekly[["R_score","F_score","M_score"]].mean(axis=1).round(2)

display(weekly[[
    "stockcode","week_start",
    "avg_customer_prior_recency_days",
    "avg_customer_prior_frequency",
    "avg_customer_prior_monetary",
    "RFM_score"
]].head(10))


# 8.1. Lag & rolling
for lag in [1,2,4]:
    weekly[f"sales_lag_{lag}"] = g["sales"].shift(lag)

past_sales = g["sales"].shift(1)

weekly["sales_roll_mean_4"] = past_sales.groupby(
    weekly["stockcode"]
).transform(lambda s: s.rolling(4,min_periods=1).mean())

weekly["sales_roll_mean_8"] = past_sales.groupby(
    weekly["stockcode"]
).transform(lambda s: s.rolling(8,min_periods=1).mean())

weekly["sales_roll_std_4"] = past_sales.groupby(
    weekly["stockcode"]
).transform(lambda s: s.rolling(4,min_periods=2).std())

# Hoạt động thị trường/nhu cầu trong quá khứ
weekly["transaction_roll_mean_4"] = g["transaction_count"].shift(1).groupby(
    weekly["stockcode"]
).transform(lambda s: s.rolling(4,min_periods=1).mean())

weekly["customer_roll_mean_4"] = g["unique_customers"].shift(1).groupby(
    weekly["stockcode"]
).transform(lambda s: s.rolling(4,min_periods=1).mean())

display(weekly[[
    "stockcode","week_start","sales",
    "sales_lag_1","sales_lag_2","sales_lag_4",
    "sales_roll_mean_4","sales_roll_mean_8","sales_roll_std_4"
]].head(15))


# 9.1. Market activity proxy
weekly["market_activity_proxy_lag1"] = (
    weekly["transaction_count_lag_1"].fillna(0)
    + weekly["unique_customers_lag_1"].fillna(0)
)

display(weekly[[
    "stockcode","week_start",
    "transaction_count_lag_1","unique_customers_lag_1",
    "transaction_roll_mean_4","customer_roll_mean_4",
    "market_activity_proxy_lag1"
]].head(10))


# 10.1. Sanity checks
assert weekly["sales_lag_1"].equals(g["sales"].shift(1))
assert weekly["sales_lag_2"].equals(g["sales"].shift(2))
assert weekly["sales_lag_4"].equals(g["sales"].shift(4))

# Kiểm tra rolling thủ công cho một product có đủ dữ liệu
p = weekly["stockcode"].value_counts().index[0]
chk = weekly[weekly["stockcode"]==p].sort_values("week_start").copy()
for i in range(1, min(6,len(chk))):
    prev = chk.iloc[max(0,i-4):i]["sales"].mean()
    assert np.isclose(chk.iloc[i]["sales_roll_mean_4"], prev, equal_nan=True)

print("OK – lag/rolling không sử dụng sales của tuần hiện tại.")
print("OK – RFM dùng prior_sales/prior_tx_count trước giao dịch.")


# 11.1. Final feature list
final_features = [
    "year","month","quarter","week_of_year","trend_index",
    "is_year_end","retail_peak_season_proxy",
    "avg_price_lag_1","product_avg_price_hist","product_active_weeks_hist",
    "transaction_count_lag_1","unique_customers_lag_1",
    "price_change_vs_hist",
    "avg_customer_prior_recency_days","avg_customer_prior_frequency",
    "avg_customer_prior_monetary","R_score","F_score","M_score","RFM_score",
    "sales_lag_1","sales_lag_2","sales_lag_4",
    "sales_roll_mean_4","sales_roll_mean_8","sales_roll_std_4",
    "transaction_roll_mean_4","customer_roll_mean_4",
    "market_activity_proxy_lag1"
]
final_features = [c for c in final_features if c in weekly.columns]

print("Số feature:", len(final_features))
for i,c in enumerate(final_features,1):
    print(f"{i:02d}. {c}")

model_table = weekly[["stockcode","week_start","sales"] + final_features].copy()
display(model_table.head())


# 12.1. Tạo feature dictionary
dictionary_rows = [
("sales","Target","Tổng doanh thu = Σ(quantity × price) theo Product–Week.","Target, không dùng làm feature"),
("year","Time","Năm của tuần dự báo.","Biết trước"),
("month","Time","Tháng của tuần dự báo.","Biết trước"),
("quarter","Time","Quý của tuần dự báo.","Biết trước"),
("week_of_year","Time","Tuần trong năm.","Biết trước"),
("trend_index","Time","Số thứ tự tuần kể từ tuần đầu tiên.","Biết trước"),
("is_year_end","Time","Cờ tháng 11–12.","Biết trước"),
("retail_peak_season_proxy","Seasonality proxy","Proxy mùa cao điểm cuối năm; không phải ad spend.","Biết trước"),
("avg_price_lag_1","Product","Giá trung bình tuần trước.","Quá khứ"),
("product_avg_price_hist","Product","Giá trung bình lịch sử trước tuần hiện tại.","Quá khứ"),
("product_active_weeks_hist","Product","Số tuần sản phẩm đã hoạt động trước tuần hiện tại.","Quá khứ"),
("transaction_count_lag_1","Product","Số giao dịch tuần trước.","Quá khứ"),
("unique_customers_lag_1","Customer","Số khách hàng tuần trước.","Quá khứ"),
("price_change_vs_hist","Product","Giá tuần trước so với giá lịch sử.","Quá khứ"),
("avg_customer_prior_recency_days","RFM","Recency trung bình dựa trên giao dịch trước đó.","Quá khứ"),
("avg_customer_prior_frequency","RFM","Frequency trung bình: số giao dịch trước đó.","Quá khứ"),
("avg_customer_prior_monetary","RFM","Monetary trung bình: sales tích lũy trước đó.","Quá khứ"),
("R_score/F_score/M_score/RFM_score","RFM","Điểm RFM 1–5 từ lịch sử khách hàng.","Quá khứ"),
("sales_lag_1/2/4","Sales history","Sales của 1/2/4 tuần trước.","Quá khứ"),
("sales_roll_mean_4","Sales history","Trung bình sales 4 tuần trước.","Quá khứ"),
("sales_roll_mean_8","Sales history","Trung bình sales 8 tuần trước.","Quá khứ"),
("sales_roll_std_4","Sales history","Độ lệch chuẩn sales 4 tuần trước.","Quá khứ"),
("transaction_roll_mean_4","Activity proxy","Trung bình giao dịch 4 tuần trước.","Quá khứ"),
("customer_roll_mean_4","Activity proxy","Trung bình khách hàng 4 tuần trước.","Quá khứ"),
("market_activity_proxy_lag1","Marketing proxy","Tổng hợp hoạt động giao dịch + khách hàng tuần trước; không phải ad spend.","Quá khứ")
]
fd = pd.DataFrame(dictionary_rows, columns=["feature","group","meaning","allowed_time"])
display(fd)


# 13.1. Missing report
missing_report = pd.DataFrame({
    "feature": final_features,
    "missing_count": [model_table[c].isna().sum() for c in final_features],
    "missing_%": [round(model_table[c].isna().mean()*100,2) for c in final_features]
}).sort_values("missing_%", ascending=False)

display(missing_report)


# 14.1. Time split 70/15/15 ở mức Product-Week
dates = np.sort(model_table["week_start"].dropna().unique())
cut1 = dates[int(len(dates)*0.70)]
cut2 = dates[int(len(dates)*0.85)]

train = model_table[model_table["week_start"] < cut1].copy()
validation = model_table[(model_table["week_start"] >= cut1) & (model_table["week_start"] < cut2)].copy()
test = model_table[model_table["week_start"] >= cut2].copy()

split_info = pd.DataFrame({
    "split":["train","validation","test"],
    "rows":[len(train),len(validation),len(test)],
    "from":[train.week_start.min(),validation.week_start.min(),test.week_start.min()],
    "to":[train.week_start.max(),validation.week_start.max(),test.week_start.max()]
})
display(split_info)


# 15.1. Save outputs
weekly.to_csv(OUT_DIR/"tv3_product_week_features.csv", index=False, encoding="utf-8-sig")
fd.to_csv(OUT_DIR/"tv3_feature_dictionary.csv", index=False, encoding="utf-8-sig")
train.to_csv(OUT_DIR/"tv3_train.csv", index=False, encoding="utf-8-sig")
validation.to_csv(OUT_DIR/"tv3_validation.csv", index=False, encoding="utf-8-sig")
test.to_csv(OUT_DIR/"tv3_test.csv", index=False, encoding="utf-8-sig")

summary = pd.DataFrame({
    "metric":[
        "raw_rows","raw_columns","duplicate_rows","missing_customer_id",
        "missing_description","negative_quantity","cancelled_invoice_rows",
        "unique_products","unique_customers","unique_countries",
        "product_week_rows","final_feature_count"
    ],
    "value":[
        len(raw),raw.shape[1],df.duplicated().sum(),df.customer_id.isna().sum(),
        df.description.isna().sum(),(df.quantity<0).sum(),
        df.invoice.astype(str).str.upper().str.startswith("C").sum(),
        df.stockcode.nunique(),df.customer_id.nunique(),df.country.nunique(),
        len(weekly),len(final_features)
    ]
})
summary.to_csv(OUT_DIR/"tv3_summary.csv", index=False, encoding="utf-8-sig")

print("Đã lưu TV3 vào:", OUT_DIR)
print(sorted(p.name for p in OUT_DIR.iterdir()))


