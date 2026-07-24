import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris Decision Tree Lab",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 22px;
            border: 1px solid rgba(46, 125, 50, 0.28);
            background:
                radial-gradient(circle at top left, rgba(102, 187, 106, 0.22), transparent 55%),
                linear-gradient(135deg, rgba(46, 125, 50, 0.16), rgba(102, 187, 106, 0.10));
            box-shadow: 0 10px 30px rgba(46, 125, 50, 0.12);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            font-size: 2.15rem;
            background: linear-gradient(90deg, #2e7d32, #66bb6a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p { margin: 0; opacity: .82; font-size: 1rem; }
        .info-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(46, 125, 50, 0.25);
            border-radius: 16px;
            background: rgba(102, 187, 106, 0.04);
        }
        .prediction-box {
            padding: 1.15rem;
            border-radius: 18px;
            border: 1px solid rgba(46, 125, 50, 0.25);
            background: rgba(102, 187, 106, 0.05);
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(46, 125, 50, 0.22);
            padding: .75rem;
            border-radius: 16px;
            background: rgba(102, 187, 106, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(46, 125, 50, 0.15);
        }
        .small-note { opacity: .75; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class ModelConfig:
    criterion: str
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    test_size: float
    random_state: int


@st.cache_data

def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, list(iris.feature_names), list(iris.target_names)


@st.cache_resource

def train_model(config: ModelConfig):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(
        criterion=config.criterion,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return iris, model, X_train, X_test, y_train, y_test, y_pred


def make_confusion_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4.8))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual class",
        xlabel="Predicted class",
        title="Confusion Matrix",
    )

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )
    fig.tight_layout()
    return fig


def make_tree_figure(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=3,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure", fontsize=18, pad=16)
    fig.tight_layout()
    return fig


def make_importance_figure(importance_df):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = importance_df.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌿 Iris Decision Tree Lab</h1>
        <p>ทดลองสร้าง ประเมิน และใช้งานโมเดล Decision Tree สำหรับจำแนกสายพันธุ์ดอก Iris แบบโต้ตอบ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("🌿 เกี่ยวกับโมเดล")
    st.write("อัลกอริทึม: **Decision Tree**")
    st.write("Dataset: **Iris** (150 ตัวอย่าง, 3 สายพันธุ์)")
    st.divider()

    st.header("⚙️ ตั้งค่าโมเดล")
    criterion_label = st.selectbox(
        "เกณฑ์การแบ่งโหนด",
        ["Entropy / Information Gain", "Gini Impurity", "Log Loss"],
        index=0,
    )
    criterion_map = {
        "Entropy / Information Gain": "entropy",
        "Gini Impurity": "gini",
        "Log Loss": "log_loss",
    }

    unlimited_depth = st.checkbox("ไม่จำกัดความลึก", value=False)
    max_depth_value = st.slider(
        "ความลึกสูงสุดของต้นไม้",
        min_value=1,
        max_value=10,
        value=3,
        disabled=unlimited_depth,
    )
    min_samples_split = st.slider("จำนวนตัวอย่างขั้นต่ำก่อนแบ่งโหนด", 2, 20, 2)
    min_samples_leaf = st.slider("จำนวนตัวอย่างขั้นต่ำใน Leaf", 1, 10, 1)

    st.divider()
    st.header("🧪 การแบ่งข้อมูล")
    test_size_percent = st.slider("สัดส่วนข้อมูล Test (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)

    st.divider()
    st.caption("ปรับค่าแล้วผลลัพธ์จะคำนวณใหม่อัตโนมัติ")

config = ModelConfig(
    criterion=criterion_map[criterion_label],
    max_depth=None if unlimited_depth else max_depth_value,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    test_size=test_size_percent / 100,
    random_state=int(random_state),
)

# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
df, feature_names, class_names = load_data()
iris, model, X_train, X_test, y_train, y_test, y_pred = train_model(config)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------
metric_cols = st.columns(5)
metric_cols[0].metric("ข้อมูลทั้งหมด", f"{len(df)}")
metric_cols[1].metric("Train", f"{len(X_train)}")
metric_cols[2].metric("Test", f"{len(X_test)}")
metric_cols[3].metric("Accuracy", f"{accuracy:.2%}")
metric_cols[4].metric("จำนวน Leaf", f"{model.get_n_leaves()}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_overview, tab_tree, tab_eval, tab_predict, tab_data = st.tabs(
    [
        "📌 ภาพรวม",
        "🌳 โครงสร้างต้นไม้",
        "📊 ประเมินโมเดล",
        "🔮 ทดลองทำนาย",
        "🗂️ ชุดข้อมูล",
    ]
)

with tab_overview:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("ชุดข้อมูล Iris")
        st.markdown(
            """
            <div class="info-card">
            ชุดข้อมูลประกอบด้วยดอก Iris 150 ตัวอย่าง แบ่งเป็น 3 สายพันธุ์ ได้แก่
            <b>setosa</b>, <b>versicolor</b> และ <b>virginica</b> โดยใช้คุณลักษณะ 4 ตัว:
            ความยาวและความกว้างของกลีบเลี้ยง รวมถึงความยาวและความกว้างของกลีบดอก
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### จำนวนข้อมูลแต่ละสายพันธุ์")
        st.bar_chart(df["species"].value_counts())

    with right:
        st.subheader("ค่าพารามิเตอร์ที่กำลังใช้")
        parameter_df = pd.DataFrame(
            {
                "พารามิเตอร์": [
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "test_size",
                    "random_state",
                ],
                "ค่า": [
                    config.criterion,
                    "ไม่จำกัด" if config.max_depth is None else config.max_depth,
                    config.min_samples_split,
                    config.min_samples_leaf,
                    f"{config.test_size:.0%}",
                    config.random_state,
                ],
            }
        )
        st.dataframe(parameter_df, hide_index=True, use_container_width=True)

        st.markdown("#### Feature ที่สำคัญที่สุด")
        top_feature = importance_df.iloc[0]
        st.success(f"{top_feature['feature']} — importance = {top_feature['importance']:.4f}")

with tab_tree:
    st.subheader("โครงสร้าง Decision Tree")
    st.pyplot(make_tree_figure(model, feature_names, class_names), use_container_width=True)

    st.markdown("#### กฎการตัดสินใจในรูปแบบข้อความ")
    rules = export_text(model, feature_names=feature_names)
    st.code(rules, language="text")

    tree_buffer = io.BytesIO()
    make_tree_figure(model, feature_names, class_names).savefig(
        tree_buffer, format="png", dpi=180, bbox_inches="tight"
    )
    st.download_button(
        "ดาวน์โหลดภาพต้นไม้ (.png)",
        data=tree_buffer.getvalue(),
        file_name="iris_decision_tree.png",
        mime="image/png",
    )

with tab_eval:
    eval_left, eval_right = st.columns([1, 1])
    with eval_left:
        st.subheader("Confusion Matrix")
        st.pyplot(make_confusion_figure(cm, class_names), use_container_width=True)

    with eval_right:
        st.subheader("Feature Importance")
        st.pyplot(make_importance_figure(importance_df), use_container_width=True)
        st.dataframe(
            importance_df.style.format({"importance": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Classification Report")
    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(
        "Accuracy นี้วัดจากชุด Test ที่แยกออกจากข้อมูลฝึกตามสัดส่วนที่กำหนดในแถบด้านซ้าย"
    )

with tab_predict:
    st.subheader("กรอกค่าดอกไม้เพื่อทำนายสายพันธุ์")
    st.markdown(
        '<p class="small-note">หน่วยของข้อมูลทุกตัวแปรเป็นเซนติเมตร</p>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)
    with p1:
        sepal_length = st.slider(
            "Sepal length",
            float(df[feature_names[0]].min()),
            float(df[feature_names[0]].max()),
            5.1,
            0.1,
        )
        sepal_width = st.slider(
            "Sepal width",
            float(df[feature_names[1]].min()),
            float(df[feature_names[1]].max()),
            3.5,
            0.1,
        )
    with p2:
        petal_length = st.slider(
            "Petal length",
            float(df[feature_names[2]].min()),
            float(df[feature_names[2]].max()),
            1.4,
            0.1,
        )
        petal_width = st.slider(
            "Petal width",
            float(df[feature_names[3]].min()),
            float(df[feature_names[3]].max()),
            0.2,
            0.1,
        )

    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names,
    )
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    st.markdown(
        f"""
        <div class="prediction-box">
            <h3 style="margin-top:0">ผลการทำนาย: {class_names[prediction]}</h3>
            <p style="margin-bottom:0">โมเดลเลือกคลาสที่มีค่าความน่าจะเป็นสูงสุดจากเส้นทางในต้นไม้ตัดสินใจ</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {"species": class_names, "probability": probabilities}
    ).set_index("species")
    st.bar_chart(probability_df)
    st.dataframe(
        probability_df.style.format({"probability": "{:.2%}"}),
        use_container_width=True,
    )

with tab_data:
    st.subheader("ข้อมูล Iris ทั้งหมด")
    species_filter = st.multiselect(
        "เลือกสายพันธุ์ที่ต้องการแสดง",
        options=class_names,
        default=class_names,
    )
    filtered_df = df[df["species"].isin(species_filter)]
    st.dataframe(filtered_df, hide_index=True, use_container_width=True, height=430)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ดาวน์โหลดข้อมูลที่แสดง (.csv)",
        data=csv_data,
        file_name="iris_dataset.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("<p style='text-align:center;color:#5a6b8c;margin-top:30px;'>Made with Streamlit · Machine Learning Projects</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#5a6b8c;margin-top:4px;'>Develop By tpp72</p>", unsafe_allow_html=True)