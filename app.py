import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
from scipy.optimize import fsolve
import warnings
import base64
warnings.filterwarnings('ignore')

# Helper function to convert image to base64 for HTML embedding
def get_logo_base64():
    try:
        with open("watermelon-logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Page config
st.set_page_config(
    page_title="wtrmln Financial Modeling",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for watermelon branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""
        <div style="text-align: center; background: linear-gradient(135deg, #B0E0D0 50%, #ff4757 50%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <img src="data:image/png;base64,{logo_b64}" width="80" style="margin-bottom: 15px;">
            <h1 style="margin: 10px 0;">wtrmln Financial Modeling</h1>
            <p style="margin: 0; opacity: 0.9;">Advanced Project Finance & Investment Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #B0E0D0 50%, #ff4757 50%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🍉</div>
            <h1 style="margin: 10px 0;">wtrmln Financial Modeling</h1>
            <p style="margin: 0; opacity: 0.9;">Advanced Project Finance & Investment Analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("🔧 Model Parameters")

# Industry template selection
industry = st.sidebar.selectbox(
    "Industry Template",
    ["Oil & Gas Partnership", "Real Estate Development", "SaaS Acquisition", "Manufacturing Expansion", "Custom"]
)

# Project basics
st.sidebar.subheader("📅 Project Timeline")
start_date = st.sidebar.date_input("Project Start Date", datetime.now())
project_duration = st.sidebar.slider("Project Duration (Years)", 1, 10, 5)
revenue_delay = st.sidebar.slider("Revenue Delay (Months)", 0, 24, 6)

# Financial parameters based on industry
st.sidebar.subheader("💰 Financial Inputs")

if industry == "Oil & Gas Partnership":
    initial_capex = st.sidebar.number_input("Initial CapEx ($M)", 0.1, 50.0, 2.0, 0.1)
    monthly_revenue = st.sidebar.number_input("Monthly Revenue ($K)", 100, 2000, 650, 50)
    monthly_opex = st.sidebar.number_input("Monthly OpEx ($K)", 10, 500, 50, 10)
    oil_price = st.sidebar.number_input("Oil Price ($/barrel)", 30, 100, 45, 1)
    
elif industry == "Real Estate Development":
    initial_capex = st.sidebar.number_input("Development Cost ($M)", 1.0, 100.0, 15.0, 0.5)
    monthly_revenue = st.sidebar.number_input("Monthly Rental Income ($K)", 50, 1000, 200, 25)
    monthly_opex = st.sidebar.number_input("Monthly Maintenance ($K)", 5, 200, 25, 5)
    
elif industry == "SaaS Acquisition":
    initial_capex = st.sidebar.number_input("Acquisition Price ($M)", 0.5, 20.0, 5.0, 0.1)
    monthly_revenue = st.sidebar.number_input("Monthly ARR ($K)", 50, 1000, 300, 25)
    monthly_opex = st.sidebar.number_input("Monthly Costs ($K)", 20, 400, 100, 10)
    
else:  # Manufacturing or Custom
    initial_capex = st.sidebar.number_input("Initial Investment ($M)", 0.1, 100.0, 5.0, 0.1)
    monthly_revenue = st.sidebar.number_input("Monthly Revenue ($K)", 50, 2000, 400, 25)
    monthly_opex = st.sidebar.number_input("Monthly OpEx ($K)", 10, 800, 75, 10)

# CapEx staging options
capex_staging = st.sidebar.selectbox(
    "CapEx Investment Timing",
    ["Upfront (100% at start)", "Staged (50% + 50% at 6mo)", "Delayed (100% at 3mo)", "Custom Staging"]
)

# Risk parameters
st.sidebar.subheader("⚡ Risk Analysis")
discount_rate = st.sidebar.slider("Discount Rate (%)", 5, 25, 12, 1) / 100
monte_carlo_runs = st.sidebar.selectbox("Monte Carlo Simulations", [1000, 5000, 10000], index=1)
sensitivity_analysis = st.sidebar.checkbox("Enable Sensitivity Analysis", True)

# Helper functions
def create_cash_flow_schedule(start_date, duration_years, revenue_delay_months, 
                            initial_capex, monthly_rev, monthly_opex, staging):
    """Create detailed cash flow schedule with dates"""
    
    cash_flows = []
    dates = []
    
    # Create monthly dates
    for month in range(duration_years * 12):
        date = start_date + timedelta(days=30 * month)
        dates.append(date)
    
    # Add CapEx based on staging
    if staging == "Upfront (100% at start)":
        cash_flows.append(-initial_capex * 1_000_000)  # Convert to dollars
        for i in range(1, len(dates)):
            cash_flows.append(0)  # No more CapEx
    elif staging == "Staged (50% + 50% at 6mo)":
        cash_flows.append(-initial_capex * 500_000)  # 50% upfront
        for i in range(1, len(dates)):
            if i == 6:  # 6 months later
                cash_flows.append(-initial_capex * 500_000)  # Other 50%
            else:
                cash_flows.append(0)
    elif staging == "Delayed (100% at 3mo)":
        cash_flows.append(0)  # Nothing upfront
        for i in range(1, len(dates)):
            if i == 3:  # 3 months later
                cash_flows.append(-initial_capex * 1_000_000)
            else:
                cash_flows.append(0)
    
    # Add operational cash flows
    for month in range(len(dates)):
        if month >= revenue_delay_months:
            # Revenue starts after delay
            operational_cf = (monthly_rev - monthly_opex) * 1000  # Convert to dollars
        else:
            # Only OpEx during delay period
            operational_cf = -monthly_opex * 1000
        
        cash_flows[month] += operational_cf
    
    return pd.DataFrame({
        'Date': dates,
        'CashFlow': cash_flows,
        'CumulativeCF': np.cumsum(cash_flows)
    })

def calculate_xnpv(cash_flows, dates, discount_rate):
    """Calculate XNPV using actual dates"""
    start_date = dates[0]
    xnpv = 0
    
    for cf, date in zip(cash_flows, dates):
        days_diff = (date - start_date).days
        years_diff = days_diff / 365.25
        xnpv += cf / (1 + discount_rate) ** years_diff
    
    return xnpv

def calculate_xirr(cash_flows, dates, guess=0.1):
    """Calculate XIRR using Newton-Raphson method"""
    start_date = dates[0]
    
    def xnpv_func(rate):
        result = 0
        for cf, date in zip(cash_flows, dates):
            days_diff = (date - start_date).days
            years_diff = days_diff / 365.25
            result += cf / (1 + rate) ** years_diff
        return result
    
    try:
        irr = fsolve(xnpv_func, guess)[0]
        return max(-0.99, min(5.0, irr))  # Cap between -99% and 500%
    except:
        return 0

def monte_carlo_simulation(base_params, num_runs=5000):
    """Run Monte Carlo simulation with parameter uncertainty"""
    results = []
    
    for _ in range(num_runs):
        # Add uncertainty to key parameters
        capex_variation = np.random.normal(1.0, 0.15)  # ±15% uncertainty
        revenue_variation = np.random.normal(1.0, 0.20)  # ±20% uncertainty
        opex_variation = np.random.normal(1.0, 0.10)  # ±10% uncertainty
        
        # Create scenario
        scenario_capex = base_params['capex'] * capex_variation
        scenario_revenue = base_params['revenue'] * revenue_variation
        scenario_opex = base_params['opex'] * opex_variation
        
        # Calculate cash flows for this scenario
        cf_data = create_cash_flow_schedule(
            base_params['start_date'],
            base_params['duration'],
            base_params['revenue_delay'],
            scenario_capex,
            scenario_revenue,
            scenario_opex,
            base_params['staging']
        )
        
        # Calculate metrics
        xnpv = calculate_xnpv(cf_data['CashFlow'], cf_data['Date'], base_params['discount_rate'])
        xirr = calculate_xirr(cf_data['CashFlow'], cf_data['Date'])
        
        results.append({
            'XNPV': xnpv,
            'XIRR': xirr,
            'CapEx_Factor': capex_variation,
            'Revenue_Factor': revenue_variation,
            'OpEx_Factor': opex_variation
        })
    
    return pd.DataFrame(results)

# Main calculation
base_params = {
    'start_date': start_date,
    'duration': project_duration,
    'revenue_delay': revenue_delay,
    'capex': initial_capex,
    'revenue': monthly_revenue,
    'opex': monthly_opex,
    'staging': capex_staging,
    'discount_rate': discount_rate
}

# Create base case cash flows
cf_df = create_cash_flow_schedule(
    start_date, project_duration, revenue_delay,
    initial_capex, monthly_revenue, monthly_opex, capex_staging
)

# Calculate base metrics
base_xnpv = calculate_xnpv(cf_df['CashFlow'], cf_df['Date'], discount_rate)
base_xirr = calculate_xirr(cf_df['CashFlow'], cf_df['Date'])

# Payback calculation
cumulative_cf = cf_df['CumulativeCF'].values
payback_months = None
for i, cum_cf in enumerate(cumulative_cf):
    if cum_cf > 0:
        payback_months = i
        break

payback_years = payback_months / 12 if payback_months else project_duration

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "XNPV (Date-based NPV)",
        f"${base_xnpv/1_000_000:.2f}M",
        delta=f"Discount Rate: {discount_rate*100:.1f}%"
    )

with col2:
    xirr_display = base_xirr * 100 if base_xirr > -0.99 else -99
    risk_class = "risk-low" if xirr_display > 20 else "risk-medium" if xirr_display > 10 else "risk-high"
    st.metric(
        "XIRR (Date-based IRR)",
        f"{xirr_display:.1f}%",
        delta="vs. Required Return"
    )

with col3:
    st.metric(
        "Payback Period",
        f"{payback_years:.1f} years",
        delta=f"{payback_months} months" if payback_months else "No payback"
    )

with col4:
    total_investment = initial_capex * 1_000_000
    total_return = base_xnpv + total_investment
    roi_percent = (total_return / total_investment - 1) * 100 if total_investment > 0 else 0
    st.metric(
        "Total ROI",
        f"{roi_percent:.1f}%",
        delta=f"${total_return/1_000_000:.1f}M return"
    )

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Cash Flow Analysis", "🎯 Monte Carlo Risk", "📈 Sensitivity Analysis", "📋 Export Model"])

with tab1:
    st.subheader("Cash Flow Timeline")
    
    # Cash flow chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cf_df['Date'],
        y=cf_df['CashFlow']/1000,
        mode='lines+markers',
        name='Monthly Cash Flow',
        line=dict(color='#ff4757', width=3),
        hovertemplate='<b>%{x}</b><br>Cash Flow: $%{y:.0f}K<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=cf_df['Date'],
        y=cf_df['CumulativeCF']/1000,
        mode='lines',
        name='Cumulative Cash Flow',
        line=dict(color='#2c3e50', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>Cumulative: $%{y:.0f}K<extra></extra>'
    ))
    
    # Add break-even line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Break-even")
    
    fig.update_layout(
        title=f"{industry} - Cash Flow Projection",
        xaxis_title="Date",
        yaxis_title="Cash Flow ($K)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cash flow summary table
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Summary")
        summary_data = {
            "Metric": [
                "Total Investment",
                "Total Revenue",
                "Total OpEx",
                "Net Cash Flow",
                "Revenue Delay",
                "Investment Period"
            ],
            "Value": [
                f"${initial_capex:.1f}M",
                f"${cf_df['CashFlow'][cf_df['CashFlow'] > 0].sum()/1_000_000:.1f}M",
                f"${abs(cf_df['CashFlow'][cf_df['CashFlow'] < 0].sum())/1_000_000:.1f}M",
                f"${cf_df['CashFlow'].sum()/1_000_000:.1f}M",
                f"{revenue_delay} months",
                f"{project_duration} years"
            ]
        }
        st.dataframe(summary_data, use_container_width=True)
    
    with col2:
        st.subheader("Key Dates")
        revenue_start_date = start_date + timedelta(days=30 * revenue_delay)
        project_end_date = start_date + timedelta(days=365 * project_duration)
        
        date_data = {
            "Event": [
                "Project Start",
                "Revenue Begins",
                "Payback Achieved",
                "Project End"
            ],
            "Date": [
                start_date.strftime("%b %Y"),
                revenue_start_date.strftime("%b %Y"),
                (start_date + timedelta(days=30 * payback_months)).strftime("%b %Y") if payback_months else "Never",
                project_end_date.strftime("%b %Y")
            ]
        }
        st.dataframe(date_data, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Risk Analysis")
    
    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {monte_carlo_runs:,} simulations..."):
            mc_results = monte_carlo_simulation(base_params, monte_carlo_runs)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # XNPV distribution
            fig_npv = px.histogram(
                mc_results, 
                x='XNPV',
                title='XNPV Distribution',
                color_discrete_sequence=['#ff4757']
            )
            fig_npv.add_vline(x=base_xnpv, line_dash="dash", line_color="black", annotation_text="Base Case")
            fig_npv.update_xaxis(title="XNPV ($)")
            st.plotly_chart(fig_npv, use_container_width=True)
        
        with col2:
            # XIRR distribution
            fig_irr = px.histogram(
                mc_results, 
                x='XIRR',
                title='XIRR Distribution',
                color_discrete_sequence=['#2c3e50']
            )
            fig_irr.add_vline(x=base_xirr, line_dash="dash", line_color="black", annotation_text="Base Case")
            fig_irr.update_xaxis(title="XIRR (%)")
            st.plotly_chart(fig_irr, use_container_width=True)
        
        # Risk metrics
        st.subheader("Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prob_positive = (mc_results['XNPV'] > 0).mean() * 100
            st.metric("Probability of Positive NPV", f"{prob_positive:.1f}%")
        
        with col2:
            percentile_5 = np.percentile(mc_results['XNPV'], 5)
            st.metric("5th Percentile NPV", f"${percentile_5/1_000_000:.1f}M")
        
        with col3:
            percentile_95 = np.percentile(mc_results['XNPV'], 95)
            st.metric("95th Percentile NPV", f"${percentile_95/1_000_000:.1f}M")
        
        with col4:
            value_at_risk = base_xnpv - percentile_5
            st.metric("Value at Risk (95%)", f"${value_at_risk/1_000_000:.1f}M")

with tab3:
    if sensitivity_analysis:
        st.subheader("Sensitivity Analysis")
        
        # Create sensitivity ranges
        sensitivity_ranges = {
            'CapEx': np.linspace(0.7, 1.3, 7),
            'Revenue': np.linspace(0.7, 1.3, 7),
            'OpEx': np.linspace(0.7, 1.3, 7),
            'Discount Rate': np.linspace(0.5, 2.0, 7)
        }
        
        sensitivity_results = {}
        
        for param, multipliers in sensitivity_ranges.items():
            npvs = []
            
            for mult in multipliers:
                if param == 'CapEx':
                    cf_temp = create_cash_flow_schedule(
                        start_date, project_duration, revenue_delay,
                        initial_capex * mult, monthly_revenue, monthly_opex, capex_staging
                    )
                elif param == 'Revenue':
                    cf_temp = create_cash_flow_schedule(
                        start_date, project_duration, revenue_delay,
                        initial_capex, monthly_revenue * mult, monthly_opex, capex_staging
                    )
                elif param == 'OpEx':
                    cf_temp = create_cash_flow_schedule(
                        start_date, project_duration, revenue_delay,
                        initial_capex, monthly_revenue, monthly_opex * mult, capex_staging
                    )
                else:  # Discount Rate
                    cf_temp = cf_df.copy()
                
                if param == 'Discount Rate':
                    npv = calculate_xnpv(cf_temp['CashFlow'], cf_temp['Date'], discount_rate * mult)
                else:
                    npv = calculate_xnpv(cf_temp['CashFlow'], cf_temp['Date'], discount_rate)
                
                npvs.append(npv)
            
            sensitivity_results[param] = {
                'multipliers': multipliers,
                'npvs': npvs
            }
        
        # Tornado chart
        fig_tornado = go.Figure()
        
        colors = ['#ff4757', '#2c3e50', '#f39c12', '#27ae60']
        
        for i, (param, data) in enumerate(sensitivity_results.items()):
            fig_tornado.add_trace(go.Scatter(
                x=[(m-1)*100 for m in data['multipliers']],
                y=[npv/1_000_000 for npv in data['npvs']],
                mode='lines+markers',
                name=param,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig_tornado.add_hline(y=base_xnpv/1_000_000, line_dash="dash", line_color="gray")
        fig_tornado.update_layout(
            title="Sensitivity Analysis - XNPV vs Parameter Changes",
            xaxis_title="Parameter Change (%)",
            yaxis_title="XNPV ($M)",
            height=500
        )
        
        st.plotly_chart(fig_tornado, use_container_width=True)

with tab4:
    st.subheader("Export Financial Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Options:**")
        
        # Prepare export data
        export_df = cf_df.copy()
        export_df['Month'] = range(1, len(export_df) + 1)
        export_df['CashFlow_K'] = export_df['CashFlow'] / 1000
        export_df['CumulativeCF_K'] = export_df['CumulativeCF'] / 1000
        
        # Add summary metrics
        summary_df = pd.DataFrame({
            'Metric': ['XNPV ($M)', 'XIRR (%)', 'Payback (Years)', 'Total ROI (%)'],
            'Value': [
                round(base_xnpv/1_000_000, 2),
                round(base_xirr*100, 1),
                round(payback_years, 1),
                round(roi_percent, 1)
            ]
        })
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Cash_Flows', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add formatting
            workbook = writer.book
            money_format = workbook.add_format({'num_format': '$#,##0'})
            percent_format = workbook.add_format({'num_format': '0.0%'})
            
        output.seek(0)
        
        st.download_button(
            label="📊 Download Excel Model",
            data=output.getvalue(),
            file_name=f"wtrmln_model_{industry.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.write("**Model Summary:**")
        st.dataframe(summary_df, use_container_width=True)
        
        st.write("**Parameters Used:**")
        params_summary = {
            "Parameter": [
                "Industry", "Start Date", "Duration", "Revenue Delay",
                "Initial CapEx", "Monthly Revenue", "Monthly OpEx", 
                "CapEx Staging", "Discount Rate"
            ],
            "Value": [
                industry, start_date.strftime("%Y-%m-%d"), f"{project_duration} years",
                f"{revenue_delay} months", f"${initial_capex:.1f}M", 
                f"${monthly_revenue:.0f}K", f"${monthly_opex:.0f}K",
                capex_staging, f"{discount_rate*100:.1f}%"
            ]
        }
        st.dataframe(params_summary, use_container_width=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; padding: 20px;">
            <img src="data:image/png;base64,{logo_b64}" width="40" style="margin-bottom: 10px;">
            <br>
            <strong>wtrmln Financial Modeling Platform</strong><br>
            Advanced project finance analysis with Monte Carlo simulations<br>
            <small> All Rights Reserved © 2025</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 20px;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🍉</div>
            <strong>wtrmln Financial Modeling Platform</strong><br>
            Advanced project finance analysis with Monte Carlo simulations<br>
            <small> All Rights Reserved © 2025</small>
        </div>
        """, unsafe_allow_html=True)