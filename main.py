# main.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Estudo Regionais - Desligamentos", layout="wide")

# ======================
# Funções utilitárias
# ======================
@st.cache_data
def carregar_base(caminho: str) -> pd.DataFrame:
    df = pd.read_excel(caminho, dtype=str)

    # Helpers de conversão
    def to_datetime_safe(s):
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

    def to_num_safe(s):
        if s is None:
            return np.nan
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Converte datas
    if "Data de Rescisão" in df.columns:
        df["Data de Rescisão"] = to_datetime_safe(df["Data de Rescisão"])
    if "Data Admissão" in df.columns:
        df["Data Admissão"] = to_datetime_safe(df["Data Admissão"])

    # Converte numéricos
    if "Tempo de Casa" in df.columns:
        df["Tempo de Casa"] = to_num_safe(df["Tempo de Casa"])
    if "Idade Período" in df.columns:
        df["Idade Período"] = to_num_safe(df["Idade Período"])

    # Sanitiza strings
    for col in [
        "Regional", "Status", "Sexo", "Raça/Cor", "Cargo", "Nível de Hierarquia",
        "Diretoria", "Código Centro Custo", "GERENTE 01", "GERENTE 02"
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Período de rescisão (Ano-Mês)
    if "Data de Rescisão" in df.columns and df["Data de Rescisão"].notna().any():
        df["Ano-Mês"] = df["Data de Rescisão"].dt.to_period("M").astype(str)

    # ===== Safras (cohort) =====
    if "Data Admissão" in df.columns and df["Data Admissão"].notna().any():
        df["Admissão YM"] = df["Data Admissão"].dt.to_period("M").astype(str)
        df["Admissão Ano"] = df["Data Admissão"].dt.year

    if "Data de Rescisão" in df.columns and df["Data de Rescisão"].notna().any():
        df["Rescisão YM"] = df["Data de Rescisão"].dt.to_period("M").astype(str)

    return df


def contar_por(df: pd.DataFrame, col: str, top=None):
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "Desligados"])
    s = df.groupby(col).size().sort_values(ascending=False)
    if top:
        s = s.head(top)
    return s.reset_index(name="Desligados")


def media_por(df: pd.DataFrame, col_group: str, col_value: str):
    if col_group not in df.columns or col_value not in df.columns:
        return pd.DataFrame(columns=[col_group, f"Média {col_value}"])
    m = (
        df.groupby(col_group)[col_value]
        .mean()
        .reset_index(name=f"Média {col_value}")
        .sort_values(f"Média {col_value}", ascending=False)
    )
    return m


def pct_share(df: pd.DataFrame, by_cols: list[str]):
    g = df.groupby(by_cols).size().reset_index(name="Desligados")
    total = g["Desligados"].sum()
    g["Participação (%)"] = (g["Desligados"] / total * 100).round(2) if total else 0.0
    return g.sort_values("Desligados", ascending=False)


def get_coluna_motivo(df: pd.DataFrame):
    """Tenta identificar a coluna de motivo caso ela exista na base usando nomes comuns."""
    candidatos = [
        "Motivo", "Motivo de Desligamento", "Motivo Desligamento",
        "Motivo Rescisão", "Motivo da Rescisão", "Motivo Demissão"
    ]
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def baixar_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ==========================
# Entrada & carregamento
# ==========================
st.title("Estudo Regionais – Desligamentos")

caminho = st.text_input("Caminho do arquivo (Excel)", value="data.xlsx")
if not caminho:
    st.stop()

try:
    tabela = carregar_base(caminho)
except Exception as e:
    st.error(f"Erro ao carregar a base: {e}")
    st.stop()

# ==========================
# Filtros (Sidebar)
# ==========================
with st.sidebar:
    st.header("Filtros")

    # Período
    if "Data de Rescisão" in tabela.columns and tabela["Data de Rescisão"].notna().any():
        min_dt = pd.to_datetime(tabela["Data de Rescisão"].min())
        max_dt = pd.to_datetime(tabela["Data de Rescisão"].max())
        dt_ini, dt_fim = st.date_input(
            "Período (Data de Rescisão)",
            value=(min_dt.date(), max_dt.date()),
        )
    else:
        dt_ini, dt_fim = None, None

    # Regionais
    regioes_opts = sorted(tabela["Regional"].dropna().unique()) if "Regional" in tabela.columns else []
    regioes_sel = st.multiselect("Regional", regioes_opts, default=regioes_opts)

    # Status
    status_opts = sorted(tabela["Status"].dropna().unique()) if "Status" in tabela.columns else []
    status_sel = st.multiselect("Status (Voluntário/Involuntário)", status_opts, default=status_opts)

    # Sexo
    sexo_opts = sorted(tabela["Sexo"].dropna().unique()) if "Sexo" in tabela.columns else []
    sexo_sel = st.multiselect("Sexo", sexo_opts, default=sexo_opts)

    # Raça/Cor
    raca_opts = sorted(tabela["Raça/Cor"].dropna().unique()) if "Raça/Cor" in tabela.columns else []
    raca_sel = st.multiselect("Raça/Cor", raca_opts, default=raca_opts)

    # Cargo
    cargo_opts = sorted(tabela["Cargo"].dropna().unique()) if "Cargo" in tabela.columns else []
    cargo_sel = st.multiselect("Cargo", cargo_opts, default=cargo_opts)

# Aplica filtros
df = tabela.copy()
if "Data de Rescisão" in df.columns and dt_ini and dt_fim:
    mask_dt = (df["Data de Rescisão"] >= pd.to_datetime(dt_ini)) & (df["Data de Rescisão"] <= pd.to_datetime(dt_fim))
    df = df[mask_dt]
if "Regional" in df.columns and len(regioes_sel) > 0:
    df = df[df["Regional"].isin(regioes_sel)]
if "Status" in df.columns and len(status_sel) > 0:
    df = df[df["Status"].isin(status_sel)]
if "Sexo" in df.columns and len(sexo_sel) > 0:
    df = df[df["Sexo"].isin(sexo_sel)]
if "Raça/Cor" in df.columns and len(raca_sel) > 0:
    df = df[df["Raça/Cor"].isin(raca_sel)]
if "Cargo" in df.columns and len(cargo_sel) > 0:
    df = df[df["Cargo"].isin(cargo_sel)]

# ==========================
# KPIs (fixos no topo)
# ==========================
st.subheader("Visão Geral – KPIs")
k1, k2, k3, k4 = st.columns(4)

total_deslig = len(df)
with k1:
    st.metric("Desligamentos (Total)", total_deslig)

if "Status" in df.columns:
    vol = int((df["Status"] == "Voluntário").sum())
    inv = int((df["Status"] == "Involuntário").sum())
else:
    vol = inv = 0

with k2:
    st.metric("Voluntários", vol)
with k3:
    st.metric("Involuntários", inv)

if "Tempo de Casa" in df.columns and df["Tempo de Casa"].notna().any():
    tempo_medio = df["Tempo de Casa"].mean()
    tempo_medio_fmt = f"{tempo_medio:.2f}"
else:
    tempo_medio_fmt = "—"

with k4:
    st.metric("Tempo médio de casa (meses)", tempo_medio_fmt)

st.divider()

# ==========================
# TABS (seções separadas)
# ==========================
tabs = st.tabs([
    "Visão Geral",
    "Regionais",
    "Status",
    "Tempo de Casa",
    "Cargos / Níveis",
    "Sazonalidade",
    "Liderança",
    "Diversidade",
    "RJ vs Outras",
    "Diretoria / Centro de Custo",
    "Motivos",
    "Safra (Admissão x Desligamento)",  # NOVA ABA
    "Exportar"
])

# --------- Visão Geral
with tabs[0]:
    st.markdown("### Participação dos desligamentos por Regional")
    if "Regional" in df.columns:
        por_regional = contar_por(df, "Regional")
        c1, c2 = st.columns([2, 1])
        with c1:
            chart = (
                alt.Chart(por_regional)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y("Regional:N", sort="-x"),
                    tooltip=["Regional", "Desligados"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)
        with c2:
            st.dataframe(por_regional, use_container_width=True)
            baixar_csv_button(por_regional, "Baixar CSV (por Regional)", "desligados_por_regional.csv")
    else:
        st.info("Coluna 'Regional' não encontrada.")

# --------- Regionais
with tabs[1]:
    st.markdown("### Ranking de desligados por Regional")
    if "Regional" in df.columns:
        por_regional = contar_por(df, "Regional")
        st.dataframe(por_regional, use_container_width=True)
    else:
        st.info("Coluna 'Regional' não encontrada.")

# --------- Status
with tabs[2]:
    st.markdown("### Composição de desligamentos por Status (por Regional)")
    if {"Regional", "Status"}.issubset(df.columns):
        comp_status = df.groupby(["Regional", "Status"]).size().reset_index(name="Desligados")
        comp_status["% Regional"] = comp_status.groupby("Regional")["Desligados"] \
            .transform(lambda x: x / x.sum() * 100).round(2)

        chart2 = (
            alt.Chart(comp_status)
            .mark_bar()
            .encode(
                x=alt.X("Regional:N", sort="-y"),
                y=alt.Y("% Regional:Q", stack="normalize"),
                color=alt.Color("Status:N"),
                tooltip=["Regional", "Status", "Desligados", "% Regional"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart2, use_container_width=True)
        with st.expander("Ver tabela"):
            st.dataframe(comp_status.sort_values(["Regional", "Desligados"], ascending=[True, False]),
                         use_container_width=True)
            baixar_csv_button(comp_status, "Baixar CSV (Status x Regional)", "status_por_regional.csv")
    else:
        st.info("Colunas 'Regional' e/ou 'Status' não encontradas.")

# --------- Tempo de Casa
with tabs[3]:
    st.markdown("### Tempo médio de casa por Regional (meses)")
    if {"Regional", "Tempo de Casa"}.issubset(df.columns) and df["Tempo de Casa"].notna().any():
        tmp = media_por(df.dropna(subset=["Tempo de Casa"]), "Regional", "Tempo de Casa")
        chart3 = (
            alt.Chart(tmp)
            .mark_bar()
            .encode(
                x=alt.X("Média Tempo de Casa:Q"),
                y=alt.Y("Regional:N", sort="-x"),
                tooltip=["Regional", "Média Tempo de Casa"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart3, use_container_width=True)
        st.dataframe(tmp, use_container_width=True)
        baixar_csv_button(tmp, "Baixar CSV (Tempo médio por Regional)", "tempo_medio_por_regional.csv")
    else:
        st.info("Dados de 'Tempo de Casa' indisponíveis.")

# --------- Cargos / Níveis
with tabs[4]:
    cA, cB = st.columns(2)

    with cA:
        st.markdown("#### Desligados por Cargo (Top 10)")
        if "Cargo" in df.columns:
            por_cargo = contar_por(df, "Cargo", top=10)
            chart4 = (
                alt.Chart(por_cargo)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y("Cargo:N", sort="-x"),
                    tooltip=["Cargo", "Desligados"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart4, use_container_width=True)
            st.dataframe(por_cargo, use_container_width=True)
            baixar_csv_button(por_cargo, "Baixar CSV (Top 10 Cargos)", "top10_cargos.csv")
        else:
            st.info("Coluna 'Cargo' não encontrada.")

    with cB:
        st.markdown("#### Desligados por Nível de Hierarquia")
        if "Nível de Hierarquia" in df.columns:
            por_nivel = contar_por(df, "Nível de Hierarquia")
            chart5 = (
                alt.Chart(por_nivel)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y("Nível de Hierarquia:N", sort="-x"),
                    tooltip=["Nível de Hierarquia", "Desligados"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart5, use_container_width=True)
            st.dataframe(por_nivel, use_container_width=True)
            baixar_csv_button(por_nivel, "Baixar CSV (Nível de Hierarquia)", "desligados_por_nivel.csv")
        else:
            st.info("Coluna 'Nível de Hierarquia' não encontrada.")

# --------- Sazonalidade
with tabs[5]:
    st.markdown("### Sazonalidade de desligamentos (por mês)")
    if "Ano-Mês" in df.columns:
        por_mes = contar_por(df, "Ano-Mês")
        chart6 = (
            alt.Chart(por_mes)
            .mark_line(point=True)
            .encode(
                x=alt.X("Ano-Mês:N", sort=por_mes["Ano-Mês"].tolist()),
                y=alt.Y("Desligados:Q"),
                tooltip=["Ano-Mês", "Desligados"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart6, use_container_width=True)
        st.dataframe(por_mes, use_container_width=True)
        baixar_csv_button(por_mes, "Baixar CSV (Sazonalidade)", "desligados_por_mes.csv")
    else:
        st.info("Sem coluna de período (Ano-Mês).")

# --------- Liderança
with tabs[6]:
    st.markdown("### Liderança – Top 10 desligamentos por GERENTE 01")
    if "GERENTE 01" in df.columns:
        por_g1 = contar_por(df, "GERENTE 01", top=10)
        chart7 = (
            alt.Chart(por_g1)
            .mark_bar()
            .encode(
                x=alt.X("Desligados:Q"),
                y=alt.Y("GERENTE 01:N", sort="-x"),
                tooltip=["GERENTE 01", "Desligados"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart7, use_container_width=True)
        st.dataframe(por_g1, use_container_width=True)
        baixar_csv_button(por_g1, "Baixar CSV (Top 10 GERENTE 01)", "top10_gerente01.csv")
    else:
        st.info("Coluna 'GERENTE 01' não encontrada.")

# --------- Diversidade
with tabs[7]:
    st.markdown("### Diversidade – composição por Sexo e Raça/Cor")
    if {"Regional", "Sexo"}.issubset(df.columns):
        st.markdown("**Sexo x Regional (Stacked)**")
        chart8a = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Regional:N"),
                y=alt.Y("count():Q", stack="normalize", title="% dentro da Regional"),
                color="Sexo:N",
                tooltip=[alt.Tooltip("count()", title="Desligados"), "Regional", "Sexo"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart8a, use_container_width=True)
    else:
        st.info("Colunas 'Regional' e/ou 'Sexo' não encontradas.")

    if {"Regional", "Raça/Cor"}.issubset(df.columns):
        st.markdown("**Raça/Cor x Regional (Heatmap de participação)**")
        raca_reg = df.groupby(["Regional", "Raça/Cor"]).size().reset_index(name="Desligados")
        raca_reg["% na Regional"] = raca_reg.groupby("Regional")["Desligados"] \
            .transform(lambda x: x / x.sum() * 100).round(2)

        chart8b = (
            alt.Chart(raca_reg)
            .mark_rect()
            .encode(
                x=alt.X("Regional:N"),
                y=alt.Y("Raça/Cor:N"),
                tooltip=["Regional", "Raça/Cor", "Desligados", "% na Regional"],
                color=alt.Color("% na Regional:Q")
            )
            .properties(height=320)
        )
        st.altair_chart(chart8b, use_container_width=True)
        with st.expander("Ver tabela"):
            st.dataframe(raca_reg.sort_values(["Regional", "Desligados"], ascending=[True, False]),
                         use_container_width=True)
            baixar_csv_button(raca_reg, "Baixar CSV (Diversidade Raça/Cor)", "diversidade_raca_regional.csv")

# --------- RJ vs Outras
with tabs[8]:
    st.markdown("### RJ vs Outras – cortes comparativos")
    if "Regional" in df.columns:
        tem_rj = (df["Regional"].str.upper() == "REGIONAL RJ").any()
        if tem_rj:
            df2 = df.copy()
            df2["Grupo Regional"] = np.where(df2["Regional"].str.upper() == "REGIONAL RJ", "RJ", "Outras")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Participação por Status (RJ vs Outras)**")
                if "Status" in df2.columns:
                    g = df2.groupby(["Grupo Regional", "Status"]).size().reset_index(name="Desligados")
                    g["% no Grupo"] = g.groupby("Grupo Regional")["Desligados"].transform(lambda x: x / x.sum() * 100).round(2)
                    chart_rj1 = (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=alt.X("Grupo Regional:N"),
                            y=alt.Y("% no Grupo:Q", stack="normalize"),
                            color="Status:N",
                            tooltip=["Grupo Regional", "Status", "Desligados", "% no Grupo"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_rj1, use_container_width=True)
                    baixar_csv_button(g, "Baixar CSV (RJ vs Outras – Status)", "rj_vs_outras_status.csv")
                else:
                    st.info("Sem coluna 'Status' para comparação.")

            with c2:
                st.markdown("**Tempo médio de casa (meses) – RJ vs Outras**")
                if "Tempo de Casa" in df2.columns and df2["Tempo de Casa"].notna().any():
                    rj_tmp = (
                        df2.dropna(subset=["Tempo de Casa"])
                        .groupby("Grupo Regional")["Tempo de Casa"]
                        .mean()
                        .reset_index(name="Tempo médio (meses)")
                    )
                    chart_rj2 = (
                        alt.Chart(rj_tmp)
                        .mark_bar()
                        .encode(
                            x=alt.X("Grupo Regional:N"),
                            y=alt.Y("Tempo médio (meses):Q"),
                            tooltip=["Grupo Regional", "Tempo médio (meses)"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_rj2, use_container_width=True)
                    baixar_csv_button(rj_tmp, "Baixar CSV (RJ vs Outras – Tempo de Casa)", "rj_vs_outras_tempo_casa.csv")
                else:
                    st.info("Sem dados suficientes de 'Tempo de Casa'.")
        else:
            st.info("Não encontrei 'REGIONAL RJ' (exatamente) na base. Verifique a grafia.")
    else:
        st.info("Coluna 'Regional' não encontrada.")

# --------- Diretoria / Centro de Custo
with tabs[9]:
    st.markdown("### Diretoria e Centro de Custo")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Desligados por Diretoria**")
        if "Diretoria" in df.columns:
            por_dir = contar_por(df, "Diretoria")
            chart_dir = (
                alt.Chart(por_dir)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y("Diretoria:N", sort="-x"),
                    tooltip=["Diretoria", "Desligados"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart_dir, use_container_width=True)
            st.dataframe(por_dir, use_container_width=True)
            baixar_csv_button(por_dir, "Baixar CSV (Diretoria)", "desligados_por_diretoria.csv")
        else:
            st.info("Coluna 'Diretoria' não encontrada.")

    with c2:
        st.markdown("**Desligados por Centro de Custo**")
        if "Código Centro Custo" in df.columns:
            por_cc = contar_por(df, "Código Centro Custo")
            chart_cc = (
                alt.Chart(por_cc)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y("Código Centro Custo:N", sort="-x"),
                    tooltip=["Código Centro Custo", "Desligados"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart_cc, use_container_width=True)
            st.dataframe(por_cc, use_container_width=True)
            baixar_csv_button(por_cc, "Baixar CSV (Centro de Custo)", "desligados_por_cc.csv")
        else:
            st.info("Coluna 'Código Centro Custo' não encontrada.")

    st.markdown("**Heatmap Diretoria x Regional**")
    if {"Diretoria", "Regional"}.issubset(df.columns):
        heat = df.groupby(["Diretoria", "Regional"]).size().reset_index(name="Desligados")
        chart_heat = (
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X("Regional:N"),
                y=alt.Y("Diretoria:N"),
                color=alt.Color("Desligados:Q"),
                tooltip=["Diretoria", "Regional", "Desligados"]
            )
            .properties(height=380)
        )
        st.altair_chart(chart_heat, use_container_width=True)
        with st.expander("Ver tabela"):
            st.dataframe(heat.sort_values(["Diretoria", "Desligados"], ascending=[True, False]), use_container_width=True)
            baixar_csv_button(heat, "Baixar CSV (Heatmap Diretoria x Regional)", "heatmap_diretoria_regional.csv")
    else:
        st.info("Faltam colunas para o heatmap (Diretoria/Regional).")

# --------- Motivos
with tabs[10]:
    st.markdown("### Motivos de desligamento")
    motivo_col = get_coluna_motivo(df)
    if motivo_col:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Motivos (geral)**")
            top_motivo = contar_por(df, motivo_col, top=15)
            chart_mot = (
                alt.Chart(top_motivo)
                .mark_bar()
                .encode(
                    x=alt.X("Desligados:Q"),
                    y=alt.Y(f"{motivo_col}:N", sort="-x"),
                    tooltip=[motivo_col, "Desligados"]
                )
                .properties(height=380)
            )
            st.altair_chart(chart_mot, use_container_width=True)
            st.dataframe(top_motivo, use_container_width=True)
            baixar_csv_button(top_motivo, "Baixar CSV (Top Motivos)", "top_motivos.csv")

        with c2:
            st.markdown("**Motivos por Status**")
            if "Status" in df.columns:
                mot_status = df.groupby([motivo_col, "Status"]).size().reset_index(name="Desligados")
                chart_ms = (
                    alt.Chart(mot_status)
                    .mark_bar()
                    .encode(
                        x=alt.X("Status:N"),
                        y=alt.Y("Desligados:Q"),
                        color="Status:N",
                        column=alt.Column(f"{motivo_col}:N", header=alt.Header(labelAngle=-30)),
                        tooltip=[motivo_col, "Status", "Desligados"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_ms, use_container_width=True)
                with st.expander("Ver tabela"):
                    st.dataframe(mot_status.sort_values(["Status", "Desligados"], ascending=[True, False]),
                                 use_container_width=True)
                    baixar_csv_button(mot_status, "Baixar CSV (Motivos por Status)", "motivos_por_status.csv")
            else:
                st.info("Sem 'Status' para detalhar motivos por tipo.")
    else:
        st.info("Não encontrei coluna de **motivo**. Se existir, renomeie para 'Motivo' (ou um dos nomes comuns) ou informe o nome para eu ajustar no código.")

# --------- Safra (Admissão x Desligamento)
with tabs[11]:
    st.markdown("### Safra de Admissão × Mês de Desligamento (Cohort)")
    cols_ok = {"Admissão YM", "Rescisão YM", "Data Admissão", "Data de Rescisão"}
    if cols_ok.issubset(df.columns) and df["Admissão YM"].notna().any() and df["Rescisão YM"].notna().any():
        base = df.dropna(subset=["Admissão YM", "Rescisão YM"]).copy()

        # Heatmap de contagem
        cohort = base.groupby(["Admissão YM", "Rescisão YM"]).size().reset_index(name="Desligados")
        st.markdown("**Contagem de desligados por safra (linhas) e mês de desligamento (colunas)**")
        chart_cohort = (
            alt.Chart(cohort)
            .mark_rect()
            .encode(
                x=alt.X("Rescisão YM:N", title="Mês de desligamento"),
                y=alt.Y("Admissão YM:N", title="Safra de admissão"),
                color=alt.Color("Desligados:Q"),
                tooltip=["Admissão YM", "Rescisão YM", "Desligados"]
            )
            .properties(height=480)
        )
        st.altair_chart(chart_cohort, use_container_width=True)

        with st.expander("Ver tabela (contagem) / Baixar CSV"):
            pivot_contagem = cohort.pivot(index="Admissão YM", columns="Rescisão YM", values="Desligados").fillna(0).astype(int)
            st.dataframe(pivot_contagem, use_container_width=True)
            baixar_csv_button(cohort, "Baixar CSV (Cohort - Contagem)", "cohort_contagem.csv")

        st.divider()

        # Heatmap percentual por safra (normaliza cada linha)
        cohort_pct = cohort.copy()
        total_por_safra = cohort_pct.groupby("Admissão YM")["Desligados"].transform("sum")
        cohort_pct["% na Safra"] = np.where(total_por_safra > 0, (cohort_pct["Desligados"] / total_por_safra * 100), 0).round(2)

        st.markdown("**Distribuição % dos desligamentos dentro de cada safra**")
        chart_cohort_pct = (
            alt.Chart(cohort_pct)
            .mark_rect()
            .encode(
                x=alt.X("Rescisão YM:N", title="Mês de desligamento"),
                y=alt.Y("Admissão YM:N", title="Safra de admissão"),
                color=alt.Color("% na Safra:Q"),
                tooltip=["Admissão YM", "Rescisão YM", "Desligados", "% na Safra"]
            )
            .properties(height=480)
        )
        st.altair_chart(chart_cohort_pct, use_container_width=True)

        with st.expander("Ver tabela (percentual por safra) / Baixar CSV"):
            pivot_pct = cohort_pct.pivot(index="Admissão YM", columns="Rescisão YM", values="% na Safra").fillna(0)
            st.dataframe(pivot_pct, use_container_width=True)
            baixar_csv_button(cohort_pct, "Baixar CSV (Cohort - % por Safra)", "cohort_percentual.csv")

        st.divider()

        # Tempo até desligar (meses) por safra — buckets
        st.markdown("**Tempo até o desligamento (meses) por safra – buckets**")
        base["Meses até desligar"] = ((base["Data de Rescisão"] - base["Data Admissão"]) / np.timedelta64(1, "M")).round(0)
        bins = [-1, 3, 6, 12, 24, np.inf]
        labels = ["0-3", "4-6", "7-12", "13-24", "25+"]
        base["Bucket Meses"] = pd.cut(base["Meses até desligar"], bins=bins, labels=labels)

        tempo_sf = base.groupby(["Admissão YM", "Bucket Meses"]).size().reset_index(name="Desligados")
        chart_tempo = (
            alt.Chart(tempo_sf)
            .mark_rect()
            .encode(
                x=alt.X("Bucket Meses:N", title="Meses até desligar"),
                y=alt.Y("Admissão YM:N", title="Safra de admissão"),
                color=alt.Color("Desligados:Q"),
                tooltip=["Admissão YM", "Bucket Meses", "Desligados"]
            )
            .properties(height=420)
        )
        st.altair_chart(chart_tempo, use_container_width=True)

        with st.expander("Ver tabela (tempo até desligar) / Baixar CSV"):
            pivot_tempo = tempo_sf.pivot(index="Admissão YM", columns="Bucket Meses", values="Desligados").fillna(0).astype(int)
            st.dataframe(pivot_tempo, use_container_width=True)
            baixar_csv_button(tempo_sf, "Baixar CSV (Tempo até desligar)", "safra_tempo_ate_desligar.csv")

    else:
        st.info("Para a safra/cohort, preciso de 'Data Admissão' e 'Data de Rescisão' válidas.")

# --------- Exportar
with tabs[12]:
    st.markdown("### Exportar dados filtrados")
    st.write("Baixe os dados **exatamente** com os filtros aplicados na barra lateral.")
    baixar_csv_button(df, "Baixar CSV (dados filtrados)", "desligados_filtrado.csv")

st.caption(
    "Obs.: Percentuais aqui representam **participação dentro dos desligados** (denominador = desligados). "
    "Para taxa de turnover real, precisamos de headcount/admissões do período e das regionais."
)
