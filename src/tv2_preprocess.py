"""
src/preprocess.py - Tiền xử lý dữ liệu cho dự án Predicting Product Sales (TV2).

Quy tắc (đã cập nhật theo kế hoạch nhóm + báo cáo EDA):
- Không chỉnh sửa dữ liệu gốc trong data/raw/.
- Tên cột được chuẩn hóa về chữ thường, snake_case.
- Loại giao dịch không hợp lệ: quantity > 0 VÀ price > 0
  (khớp định nghĩa trong Nhóm 7.docx mục 1).
- Tổng hợp về mức (stockcode × tuần):
    sales     = sum(quantity * price)
    total_qty = sum(quantity)
    order_count = số invoice khác nhau
    avg_price = mean(price)
    country   = mode(country)
- Chia train/validation/test theo MỐC TUẦN (không cắt giữa tuần),
  mốc mặc định 2011-10-05 / 2011-11-21 khớp báo cáo.
- Imputer, scaler, encoder và ngưỡng IQR chỉ FIT trên train.
- Validation dùng chọn mô hình/tuning; test chỉ dùng đánh giá cuối.
- Loại total_qty / avg_price / order_count khỏi feature của TV2 vì chúng
  được tính từ cùng các giao dịch tạo ra sales (rò rỉ nếu dự báo cùng tuần).
  TV3 sẽ tạo lag/rolling — lúc đó chúng mới là feature hợp lệ.

Cách dùng cho TV4/TV5 (fit lại trong từng fold CV):
    from preprocess import get_feature_types, build_preprocessor
    num_cols, cat_cols = get_feature_types(train)
    pre = build_preprocessor(num_cols, cat_cols)   # CHƯA fit
    model = Pipeline([("pre", pre), ("model", ...)])
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 1. CẤU HÌNH
# =============================================================================
RANDOM_STATE = 42

TARGET = "sales"
DATE_COL = "invoicedate"
WEEK_COL = "week_start"

# SỬA: ưu tiên %m/%d/%y vì dữ liệu gốc dùng MM/DD/YY (12/1/10 = 01/12/2010).
DATE_FORMATS = [
    "%m/%d/%y %H:%M",     # 12/1/10 8:26
    "%d/%m/%y %H:%M",     # fallback
    "%Y-%m-%d %H:%M:%S",
]

ID_COLS = ["invoice", "stockcode", "customer_id", "description"]
ROW_ID_COL = "row_id"          # vị trí dòng trong file gốc (mức dòng giao dịch)

# Các cột bị loại khỏi feature của TV2:
# - quantity/price: tạo trực tiếp ra sales ở mức dòng.
# - total_qty/avg_price/order_count: tổng hợp từ chính các giao dịch đó
#   → dùng trực tiếp là rò rỉ target khi dự báo cùng tuần. TV3 sẽ tạo lag/rolling.
LEAKAGE_COLS = ["quantity", "price", "total_qty", "avg_price", "order_count"]

# Mốc chia theo tuần, khớp mốc ngày đã dùng trong báo cáo.
# Có thể để None để tự tính theo tỷ lệ trên các tuần duy nhất.
SPLIT_CUTOFFS = ("2011-10-05", "2011-11-21")
TEST_RATIO = 0.10
VALIDATION_RATIO = 0.20

# Lọc giao dịch không hợp lệ (quantity > 0 và price > 0) — đúng định nghĩa báo cáo.
FILTER_INVALID_TRANSACTIONS = True

# Tùy chọn: lọc thêm stockcode không phải sản phẩm (POST, DOT, M, BANK CHARGES...).
# Mặc định TẮT để khớp con số "530.105 dòng hợp lệ, 3.922 mã sản phẩm" của báo cáo.
# Bật True nếu nhóm muốn dự báo thuần sản phẩm (loại phí ship/điều chỉnh).
FILTER_NON_PRODUCT_STOCKCODES = False
PRODUCT_STOCKCODE_REGEX = r"^\d{5}"

# Chỉ điền các cột số thực sự cần xử lý outlier sau khi xem EDA.
# Không đưa TARGET hoặc cột ID vào đây.
OUTLIER_COLS = []
IQR_K = 1.5

QTY_COL = "quantity"
PRICE_COL = "price"

RAW_PATH = Path("data/raw/dataset.csv")
PROCESSED_DIR = Path("data/processed")


# =============================================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# =============================================================================
def load_data(path=RAW_PATH):
    """Đọc CSV; không chỉnh sửa file dữ liệu gốc."""
    return pd.read_csv(
        path,
        encoding="cp1252",
        dtype={"Invoice": str, "StockCode": str},
    )


def standardize_columns(df):
    """Chuẩn hóa tên cột thành chữ thường, snake_case."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Có tên cột bị trùng sau khi chuẩn hóa: {duplicated}")
    return df


def summarize(df):
    """In kích thước, kiểu dữ liệu, số lượng thiếu/trùng và số giá trị duy nhất."""
    print(f"Kích thước dữ liệu: {df.shape}")
    print(f"Số dòng trùng hoàn toàn: {df.duplicated().sum()}")

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    print(info.to_string())
    return info


# =============================================================================
# 3. NGÀY
# =============================================================================
def parse_dates(df, date_col=DATE_COL, formats=DATE_FORMATS):
    """Chuyển cột ngày sang datetime, thử lần lượt các định dạng trong DATE_FORMATS."""
    df = df.copy()
    if not date_col or date_col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df

    raw = df[date_col]
    parsed = None
    for fmt in formats:
        candidate = pd.to_datetime(raw, format=fmt, errors="coerce")
        if candidate.notna().all():
            parsed = candidate
            print(f"Định dạng ngày đã dùng: {fmt}")
            break

    if parsed is None:
        print(
            "Cảnh báo: không định dạng nào trong DATE_FORMATS đọc được toàn bộ "
            "cột ngày; dùng đọc tự động (chậm hơn, có thể sai ngày/tháng)."
        )
        parsed = pd.to_datetime(raw, errors="coerce")

    df[date_col] = parsed
    n_invalid = int(parsed.isna().sum())
    if n_invalid:
        print(f"Cảnh báo: {n_invalid} dòng có ngày thiếu/không hợp lệ.")
    return df


def drop_invalid_dates(df, date_col=DATE_COL):
    """Bỏ dòng không có ngày hợp lệ (không thể xếp vào trục thời gian)."""
    if not date_col or date_col not in df.columns:
        return df
    before = len(df)
    df = df.dropna(subset=[date_col]).reset_index(drop=True)
    if before != len(df):
        print(f"Bỏ dòng ngày không hợp lệ: {before} -> {len(df)}.")
    return df


# =============================================================================
# 4. TARGET Ở MỨC DÒNG
# =============================================================================
def create_target(df):
    """Tạo sales = quantity * price ở mức dòng giao dịch."""
    df = df.copy()
    for col in (QTY_COL, PRICE_COL):
        if col not in df.columns:
            raise KeyError(f"Thiếu cột '{col}' để tạo {TARGET}.")
    df[TARGET] = (
        pd.to_numeric(df[QTY_COL], errors="coerce")
        * pd.to_numeric(df[PRICE_COL], errors="coerce")
    )
    return df


# =============================================================================
# 5. LÀM SẠCH Ở MỨC DÒNG
# =============================================================================
def drop_duplicates(df):
    """Xóa các dòng trùng hoàn toàn (không tính cột row_id)."""
    before = len(df)
    subset = [c for c in df.columns if c != ROW_ID_COL]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    print(f"Xóa dòng trùng: {before} -> {len(df)} (đã xóa {before-len(df)}).")
    return df


def drop_missing_target(df, target=TARGET):
    """Loại dòng không có nhãn mục tiêu vì không thể dùng để huấn luyện."""
    if target not in df.columns:
        raise KeyError(
            f"Không tìm thấy cột mục tiêu '{target}'. "
            "Hãy kiểm tra tên cột hoặc thống nhất cách tạo sales với TV1."
        )
    before = len(df)
    df = df.dropna(subset=[target]).copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    print(
        f"Bỏ dòng thiếu/không hợp lệ ở {target}: "
        f"{before} -> {len(df)} (đã bỏ {before-len(df)})."
    )
    return df


def report_anomalies(df):
    """
    Chỉ THỐNG KÊ các nhóm dòng bất thường TRƯỚC khi lọc; không xóa gì.
    Số liệu này dùng cho báo cáo mục 3.
    """
    print("\n----- THỐNG KÊ DÒNG BẤT THƯỜNG (trước khi lọc) -----")
    if "invoice" in df.columns:
        cancel = df["invoice"].astype(str).str.upper().str.startswith("C")
        print(f"Hóa đơn hủy (invoice bắt đầu bằng 'C'): {int(cancel.sum())}")
    if QTY_COL in df.columns:
        qty = pd.to_numeric(df[QTY_COL], errors="coerce")
        print(f"Dòng quantity < 0: {int((qty < 0).sum())}")
        print(f"Dòng quantity == 0: {int((qty == 0).sum())}")
    if PRICE_COL in df.columns:
        price = pd.to_numeric(df[PRICE_COL], errors="coerce")
        print(f"Dòng price < 0: {int((price < 0).sum())}")
        print(f"Dòng price == 0: {int((price == 0).sum())}")
    if "stockcode" in df.columns:
        non_product = ~df["stockcode"].astype(str).str.match(r"^\d{5}")
        print(f"Dòng có stockcode không dạng mã sản phẩm: {int(non_product.sum())}")
        if non_product.any():
            top = df.loc[non_product, "stockcode"].value_counts().head(8)
            print("  Các mã phổ biến nhất: " + ", ".join(
                f"{k} ({v})" for k, v in top.items()))
    print("------------------------------------------------------\n")


def filter_invalid_transactions(df, qty_col=QTY_COL, price_col=PRICE_COL):
    """
    Lọc bỏ giao dịch không hợp lệ: quantity <= 0 hoặc price <= 0.
    Khớp định nghĩa trong Nhóm 7.docx mục 1.
    """
    before = len(df)
    qty = pd.to_numeric(df[qty_col], errors="coerce")
    price = pd.to_numeric(df[price_col], errors="coerce")
    mask = (qty > 0) & (price > 0)
    df = df.loc[mask].reset_index(drop=True)
    print(
        f"Lọc giao dịch không hợp lệ ({qty_col}>0 & {price_col}>0): "
        f"{before} -> {len(df)} (bỏ {before-len(df)})."
    )
    return df


def filter_non_product_stockcodes(df, regex=PRODUCT_STOCKCODE_REGEX):
    """
    Lọc các dòng có stockcode không phải mã sản phẩm (POST, DOT, M, ...).
    Mặc định pipeline không gọi hàm này (FILTER_NON_PRODUCT_STOCKCODES=False).
    """
    before = len(df)
    codes = df["stockcode"].astype(str).str.strip()
    mask = codes.str.match(regex)
    df = df.loc[mask].reset_index(drop=True)
    print(
        f"Lọc stockcode không phải sản phẩm: "
        f"{before} -> {len(df)} (bỏ {before-len(df)})."
    )
    return df


# =============================================================================
# 6. TỔNG HỢP VỀ MỨC (stockcode × tuần)
# =============================================================================
def aggregate_weekly(df, date_col=DATE_COL, week_col=WEEK_COL):
    """
    Gộp dữ liệu mức dòng về mức (stockcode × tuần).
    Chỉ giữ các cặp (stockcode, tuần) CÓ phát sinh giao dịch (không điền 0),
    khớp báo cáo (101.579 quan sát trên 53 tuần cho 3.922 sản phẩm).
    """
    df = df.copy()
    # to_period("W") mặc định là W-SUN → start_time là thứ Hai đầu tuần.
    df[week_col] = df[date_col].dt.to_period("W").dt.start_time

    agg = (
        df.groupby(["stockcode", week_col], as_index=False)
        .agg(
            sales=(TARGET, "sum"),
            total_qty=(QTY_COL, "sum"),
            order_count=("invoice", "nunique"),
            avg_price=(PRICE_COL, "mean"),
            country=("country", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        )
    )

    print(
        f"Tổng hợp (stockcode × tuần): "
        f"{len(df)} dòng giao dịch -> {len(agg)} quan sát tuần "
        f"({agg['stockcode'].nunique()} sản phẩm, {agg[week_col].nunique()} tuần)."
    )
    print(f"Tổng doanh thu sau tổng hợp: {agg[TARGET].sum():,.2f}")
    return agg


# =============================================================================
# 7. CHIA TRAIN / VALIDATION / TEST THEO TUẦN
# =============================================================================
def compute_week_cutoffs(weeks, cutoffs, validation_ratio, test_ratio):
    """Trả về (val_start_week, test_start_week) ở mức tuần."""
    if cutoffs is not None:
        val_start, test_start = (pd.Timestamp(c).normalize() for c in cutoffs)
        # Đưa về đầu tuần (thứ Hai) để không cắt giữa tuần.
        val_start = val_start.to_period("W").start_time
        test_start = test_start.to_period("W").start_time
    else:
        weeks_sorted = np.sort(np.asarray(weeks))
        n = len(weeks_sorted)
        val_start = weeks_sorted[int(n * (1 - validation_ratio - test_ratio))]
        test_start = weeks_sorted[int(n * (1 - test_ratio))]
    if not val_start < test_start:
        raise ValueError("Mốc validation phải sớm hơn mốc test.")
    return val_start, test_start


def split_weekly(
    df,
    week_col=WEEK_COL,
    cutoffs=SPLIT_CUTOFFS,
    test_ratio=TEST_RATIO,
    validation_ratio=VALIDATION_RATIO,
):
    """
    Chia dữ liệu tuần thành train / validation / test theo MỐC TUẦN.
    Không có tuần nào bị cắt đôi giữa hai tập.
    """
    if test_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("test_ratio và validation_ratio phải lớn hơn 0.")
    if test_ratio + validation_ratio >= 1:
        raise ValueError("Tổng validation_ratio + test_ratio phải nhỏ hơn 1.")

    df = df.copy()
    val_start, test_start = compute_week_cutoffs(
        df[week_col].unique(), cutoffs, validation_ratio, test_ratio
    )

    train = df[df[week_col] < val_start].copy()
    validation = df[(df[week_col] >= val_start) & (df[week_col] < test_start)].copy()
    test = df[df[week_col] >= test_start].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Mốc chia làm một tập rỗng; kiểm tra SPLIT_CUTOFFS.")

    print("Chia theo mốc tuần (quá khứ -> tương lai).")
    print(f"Train: {train[week_col].min().date()} -> {train[week_col].max().date()}")
    print(
        f"Validation bắt đầu tuần: {val_start.date()} "
        f"({validation[week_col].min().date()} -> {validation[week_col].max().date()})"
    )
    print(
        f"Test bắt đầu tuần: {test_start.date()} "
        f"({test[week_col].min().date()} -> {test[week_col].max().date()})"
    )

    total = len(train) + len(validation) + len(test)
    print(
        f"Train: {len(train)} ({len(train)/total:.1%}) | "
        f"Validation: {len(validation)} ({len(validation)/total:.1%}) | "
        f"Test: {len(test)} ({len(test)/total:.1%})"
    )
    meta = {
        "method": "time_cutoff_weekly",
        "validation_start_week": str(val_start),
        "test_start_week": str(test_start),
        "cutoffs_input": list(cutoffs) if cutoffs else None,
        "n_weeks": {
            "train": int(train[week_col].nunique()),
            "validation": int(validation[week_col].nunique()),
            "test": int(test[week_col].nunique()),
        },
    }
    return train, validation, test, meta


def build_split_index(train, validation, test, week_col=WEEK_COL):
    """Bảng chỉ mục chia ở mức tuần: (stockcode, week_start) -> split."""
    parts = []
    for name, part in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        parts.append(
            pd.DataFrame({
                "stockcode": part["stockcode"].values,
                week_col: part[week_col].values,
                "split": name,
            })
        )
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 8. OUTLIER: FIT TRAIN, TRANSFORM CẢ BA TẬP
# =============================================================================
def fit_iqr_bounds(train_df, cols, k=IQR_K):
    """Tính ngưỡng IQR chỉ từ train."""
    bounds = {}
    for col in cols:
        if col not in train_df.columns:
            print(f"Cảnh báo: bỏ qua cột outlier không tồn tại: {col}")
            continue
        values = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if values.empty:
            print(f"Cảnh báo: cột {col} không có giá trị số để tính IQR.")
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_iqr_bounds(df, bounds):
    """Clip theo ngưỡng đã fit từ train; không tính lại ngưỡng."""
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(
                lower=lower, upper=upper
            )
    return df


# =============================================================================
# 9. PIPELINE: IMPUTER + SCALER + ONE-HOT ENCODER
# =============================================================================
def get_feature_types(
    df,
    target=TARGET,
    id_cols=ID_COLS,
    date_col=DATE_COL,
    week_col=WEEK_COL,
):
    """Tách biến số và biến phân loại; loại target, ID, cột ngày/tuần, cột rò rỉ."""
    excluded = (
        set(id_cols or [])
        | set(LEAKAGE_COLS)
        | {ROW_ID_COL, target}
    )
    if date_col:
        excluded.add(date_col)
    if week_col:
        excluded.add(week_col)

    feature_df = df.drop(
        columns=[c for c in excluded if c in df.columns]
    )
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Tạo bộ tiền xử lý CHƯA fit (dùng được trong Pipeline/cross-validation):
    - Biến số: median imputation -> StandardScaler.
    - Biến phân loại: most_frequent imputation -> OneHotEncoder
      (handle_unknown="ignore" để danh mục mới ở validation/test không gây lỗi).
    """
    transformers = []
    if num_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, cat_cols))
    if not transformers:
        raise ValueError("Không tìm thấy cột đặc trưng hợp lệ để tiền xử lý.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_features(preprocessor, df, feature_cols):
    """Transform dữ liệu và trả về DataFrame có tên cột sau encoding."""
    x = df.reindex(columns=feature_cols)
    transformed = preprocessor.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=names, index=df.index)


# =============================================================================
# 10. LUỒNG CHẠY CHÍNH
# =============================================================================
def run_preprocessing(path=RAW_PATH, out_dir=PROCESSED_DIR):
    """
    Chạy toàn bộ quy trình và lưu vào out_dir:
    - cleaned_lines.csv: dữ liệu mức dòng sau khi lọc (để TV3 tham khảo).
    - train.csv, validation.csv, test.csv: dữ liệu tuần sau tổng hợp, đã chia.
    - split_index.csv: (stockcode, week_start) -> split.
    - X_*.csv, y_*.csv: ma trận đặc trưng và target cho từng tập.
    - preprocessor.joblib: bộ tiền xử lý đã fit trên train.
    - preprocessing_steps.csv, preprocessing_info.json.

    Trả về train, validation, test, preprocessor (mức tuần).
    """
    steps = []

    # ----- Đọc và chuẩn hóa -----
    df = load_data(path)
    df = standardize_columns(df)
    steps.append(("Đọc dữ liệu gốc", len(df)))

    print("\n===== KIỂM TRA DỮ LIỆU BAN ĐẦU =====")
    summarize(df)

    df.insert(0, ROW_ID_COL, np.arange(len(df)))

    # ----- Ngày -----
    df = parse_dates(df)
    df = drop_invalid_dates(df)
    steps.append(("Bỏ dòng ngày không hợp lệ", len(df)))

    # ----- Target mức dòng + xóa trùng -----
    df = create_target(df)
    df = drop_duplicates(df)
    steps.append(("Xóa dòng trùng hoàn toàn", len(df)))

    # ----- Báo cáo bất thường (trước lọc) -----
    report_anomalies(df)

    # ----- Lọc giao dịch không hợp lệ (quantity > 0 và price > 0) -----
    if FILTER_INVALID_TRANSACTIONS:
        df = filter_invalid_transactions(df)
        steps.append(("Lọc giao dịch không hợp lệ (qty>0 & price>0)", len(df)))

    # ----- (Tùy chọn) Lọc stockcode không phải sản phẩm -----
    if FILTER_NON_PRODUCT_STOCKCODES:
        df = filter_non_product_stockcodes(df)
        steps.append(("Lọc stockcode không phải sản phẩm", len(df)))

    df = drop_missing_target(df)
    steps.append(("Bỏ dòng thiếu sales", len(df)))

    # Lưu mức dòng đã làm sạch (cho TV3 tham khảo / làm feature khách hàng).
    cleaned_lines = df.copy()

    # ----- Tổng hợp về mức (stockcode × tuần) -----
    weekly = aggregate_weekly(df)
    steps.append(("Tổng hợp (stockcode × tuần)", len(weekly)))

    # ----- Chia theo mốc tuần -----
    train, validation, test, split_meta = split_weekly(weekly)

    # ----- Outlier (fit trên train) -----
    outlier_cols = [
        c for c in OUTLIER_COLS
        if c in train.columns and c != TARGET
    ]
    bounds = fit_iqr_bounds(train, outlier_cols)
    train = apply_iqr_bounds(train, bounds)
    validation = apply_iqr_bounds(validation, bounds)
    test = apply_iqr_bounds(test, bounds)

    # ----- Xác định feature + fit preprocessor -----
    num_cols, cat_cols = get_feature_types(train)
    feature_cols = num_cols + cat_cols
    print(f"\nFeature số         : {num_cols}")
    print(f"Feature phân loại  : {cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(train[feature_cols])  # FIT DUY NHẤT trên train

    X_train = transform_features(preprocessor, train, feature_cols)
    X_validation = transform_features(preprocessor, validation, feature_cols)
    X_test = transform_features(preprocessor, test, feature_cols)

    y_train = train[[TARGET]].copy()
    y_validation = validation[[TARGET]].copy()
    y_test = test[[TARGET]].copy()

    # ----- Lưu -----
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_lines.to_csv(out_dir / "cleaned_lines.csv", index=False)

    train.to_csv(out_dir / "train.csv", index=False)
    validation.to_csv(out_dir / "validation.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    build_split_index(train, validation, test).to_csv(
        out_dir / "split_index.csv", index=False
    )

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_validation.to_csv(out_dir / "X_validation.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)

    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_validation.to_csv(out_dir / "y_validation.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    # Bảng số dòng trước/sau từng bước.
    steps_df = pd.DataFrame(steps, columns=["buoc", "so_dong_sau_buoc"])
    steps_df["so_dong_bi_bo"] = (
        steps_df["so_dong_sau_buoc"].shift(1) - steps_df["so_dong_sau_buoc"]
    ).fillna(0).astype(int)
    steps_df.to_csv(out_dir / "preprocessing_steps.csv", index=False)
    print("\n===== SỐ DÒNG THEO TỪNG BƯỚC =====")
    print(steps_df.to_string(index=False))

    info = {
        "target": TARGET,
        "target_formula": "sum(quantity * price) ở mức (stockcode × tuần)",
        "aggregation_level": ["stockcode", WEEK_COL],
        "date_col": DATE_COL,
        "week_col": WEEK_COL,
        "id_cols": ID_COLS,
        "leakage_cols": LEAKAGE_COLS,
        "random_state": RANDOM_STATE,
        "test_ratio": TEST_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "split": split_meta,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_columns_before_encoding": feature_cols,
        "feature_columns_after_encoding": X_train.columns.tolist(),
        "outlier_bounds_fitted_on_train": {
            col: [float(lower), float(upper)]
            for col, (lower, upper) in bounds.items()
        },
        "filter_invalid_transactions": FILTER_INVALID_TRANSACTIONS,
        "filter_non_product_stockcodes": FILTER_NON_PRODUCT_STOCKCODES,
        "steps": [{"step": s, "rows": int(n)} for s, n in steps],
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "cleaned_lines": len(cleaned_lines),
        },
    }
    with open(out_dir / "preprocessing_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=4)

    print(f"\nĐã lưu kết quả tiền xử lý tại: {out_dir.resolve()}")
    print("Đã fit preprocessor trên train và transform validation/test.")
    print("Lưu ý: sales ở mức (stockcode × tuần); total_qty/avg_price/order_count")
    print("       bị loại khỏi feature của TV2 — TV3 sẽ tạo lag/rolling.")

    return train, validation, test, preprocessor


if __name__ == "__main__":
    run_preprocessing()
