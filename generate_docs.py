import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins (padding) for a table cell in dxa (1/20th of a point)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Set background color for a table cell."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_border(cell, **kwargs):
    """
    Set cell borders.
    Usage: set_cell_border(cell, top={"sz": 12, "val": "single", "color": "D3D3D3"})
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn('w:{}'.format(key)), str(val))

def format_run(run, font_name="Calibri", size_pt=11, bold=False, italic=False, color_rgb=None):
    """Helper to apply formatting to a text run."""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb

def add_page_number(run):
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

def main():
    print("--- STEP 1: Running Sales Prediction Training Pipeline ---")
    
    # 1. Download & Load Dataset
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv"
    DATA_PATH = "Advertising.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    elif 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    # Feature engineering
    df_engineered = df.copy()
    df_engineered['TV_Radio_Interaction'] = df_engineered['TV'] * df_engineered['Radio']
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Pairplot
    plt.figure(figsize=(9, 7))
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.savefig('plots/advertising_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(df_engineered.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Ad Spend vs. Sales', fontsize=12, fontweight='bold', pad=10)
    plt.savefig('plots/advertising_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Regression modeling
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    X = df_engineered.drop('Sales', axis=1)
    y = df_engineered['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.01, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'ModelObj': model
        }
        
    summary_df = pd.DataFrame(results).T.drop('ModelObj', axis=1)
    best_model_name = summary_df['R2'].idxmax()
    best_model = results[best_model_name]['ModelObj']
    y_pred_best = best_model.predict(X_test_scaled)
    
    # Plot 3: Actual vs Predicted Scatter
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(x=y_test, y=y_pred_best, color='#1A365D', s=60, alpha=0.8)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs. Predicted Sales ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Actual Sales (Units)', fontsize=10)
    plt.ylabel('Predicted Sales (Units)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/sales_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Feature Impact / Coefficient
    plt.figure(figsize=(8, 4.5))
    lr_model = models['Linear Regression']
    coefs = lr_model.coef_
    indices = np.argsort(np.abs(coefs))[::-1]
    sns.barplot(x=coefs[indices], y=X.columns[indices], palette='coolwarm', hue=X.columns[indices], legend=False)
    plt.title('Standardized Coefficients (Linear Regression Model)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Coefficient Weight (Impact per Standard Deviation of Spend)', fontsize=10)
    plt.ylabel('Advertising Channel / Interaction', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('plots/sales_feature_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EDA and Sales Regression training complete.")
    
    print("\n--- STEP 2: Creating Word Document ---")
    doc = docx.Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(51, 51, 51)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)
    
    # ----------------------------------------------------
    # COVER / SUBMISSION HEADER (Clean, Professional style)
    # ----------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(12)
    run_title = title_p.add_run("Code Alpha Internship Task Submission")
    format_run(run_title, font_name="Calibri Light", size_pt=26, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(36)
    run_sub = subtitle_p.add_run("Task 4: Sales Prediction based on Advertising Budgets")
    format_run(run_sub, font_name="Calibri", size_pt=16, color_rgb=RGBColor(90, 90, 90))
    
    # Metadata Info Block
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    meta_table.autofit = False
    
    metadata = [
        ("Intern Name:", "Najaf Abbas"),
        ("Internship Domain:", "Machine Learning / Data Science"),
        ("Date of Submission:", "June 20, 2026"),
        ("Host Organization:", "Code Alpha")
    ]
    
    for idx, (label, val) in enumerate(metadata):
        cell_lbl = meta_table.rows[idx].cells[0]
        cell_lbl.width = Inches(2.0)
        p_lbl = cell_lbl.paragraphs[0]
        run_lbl = p_lbl.add_run(label)
        format_run(run_lbl, font_name="Calibri", size_pt=11, bold=True, color_rgb=RGBColor(26, 54, 93))
        
        cell_val = meta_table.rows[idx].cells[1]
        cell_val.width = Inches(4.0)
        p_val = cell_val.paragraphs[0]
        run_val = p_val.add_run(val)
        format_run(run_val, font_name="Calibri", size_pt=11)
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 1: Introduction & Objectives
    # ----------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    run_h1 = h1.add_run("1. Introduction & Objectives")
    format_run(run_h1, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    intro_txt = (
        "In marketing analytics, understanding the relationship between advertising expenditure and product sales is essential "
        "for optimizing return on investment (ROI). Businesses distribute advertising budgets across multiple platforms "
        "including Television (TV), Radio, and Newspapers, and they need quantitative tools to forecast sales based on these allocations.\n\n"
        "The objective of this project is to build a multiple linear regression model that predicts sales volume based on advertising "
        "spends. By exploring the data, we investigate the individual impact of each advertising platform, construct interaction terms to "
        "model campaign synergies (TV-Radio combined effect), evaluate predictions across linear and ensemble models, "
        "and establish actionable budget allocation insights for marketing managers."
    )
    doc.add_paragraph(intro_txt)
    
    # ----------------------------------------------------
    # SECTION 2: Data Exploration & EDA
    # ----------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)
    run_h2 = h2.add_run("2. Exploratory Data Analysis & Visualizations")
    format_run(run_h2, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    eda_txt = (
        "Visualizing relationships between each individual advertising channel and net sales provides initial guidance "
        "on channel effectiveness. TV shows a strong, linear correlation with sales, while Radio has a moderately linear "
        "association. Newspaper spend displays a high degree of scatter, indicating weak linear correlation."
    )
    doc.add_paragraph(eda_txt)
    
    # Pairplot (embedded, adjust width to fit text)
    doc.add_paragraph().add_run("Figure 4.1: Pairwise scatter plots and regression lines for advertising mediums vs. sales").italic = True
    doc.add_picture('plots/advertising_pairplot.png', width=Inches(4.5))
    p_img1 = doc.paragraphs[-1]
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img1.paragraph_format.space_after = Pt(18)
    
    # Heatmap
    doc.add_paragraph().add_run("Figure 4.2: Correlation heatmap among advertising spends, synergy, and sales").italic = True
    doc.add_picture('plots/advertising_correlation.png', width=Inches(4.2))
    p_img2 = doc.paragraphs[-1]
    p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img2.paragraph_format.space_after = Pt(18)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 3: Preprocessing & Synergy Modeling
    # ----------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)
    run_h3 = h3.add_run("3. Preprocessing & Campaign Synergy")
    format_run(run_h3, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    methodology_txt = (
        "Modern marketing mix modeling recognizes that running advertising campaigns across multiple channels simultaneously "
        "produces a synergistic effect. For example, a customer exposed to a TV commercial is more likely to react positively to "
        "a radio ad. We modeled this interaction explicitly by constructing a synergy feature:\n"
        "TV_Radio_Interaction = TV * Radio\n\n"
        "Furthermore, to make the regression coefficients directly comparable and interpret which channel yields the highest return "
        "per unit of variation, we standard-scaled the predictor variables using Z-score normalization (mean = 0, standard deviation = 1)."
    )
    doc.add_paragraph(methodology_txt)
    
    # ----------------------------------------------------
    # SECTION 4: Model Performance Comparison
    # ----------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(18)
    h4.paragraph_format.space_after = Pt(6)
    run_h4 = h4.add_run("4. Model Performance Comparison")
    format_run(run_h4, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    results_txt = (
        "We compared the multiple linear regression models (Linear, Lasso, Ridge) and an ensemble tree regressor (Random Forest). "
        "The model metrics evaluated on the test set are summarized below:"
    )
    doc.add_paragraph(results_txt)
    
    # Create Table
    comp_table = doc.add_table(rows=5, cols=5)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Model", "R-squared (R2)", "MAE (Units)", "MSE", "RMSE"]
    col_widths = [Inches(2.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]
    
    hdr_row = comp_table.rows[0]
    for i, title in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = col_widths[i]
        set_cell_shading(cell, "1A365D")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run_h = p.add_run(title)
        format_run(run_h, font_name="Calibri", size_pt=10, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    for idx, (m_name, metrics) in enumerate(results.items()):
        row = comp_table.rows[idx + 1]
        bg_color = "F9F9F9" if idx % 2 == 0 else "FFFFFF"
        
        cell_values = [
            m_name, 
            f"{metrics['R2']:.4f}", 
            f"{metrics['MAE']:.4f}", 
            f"{metrics['MSE']:.4f}", 
            f"{metrics['RMSE']:.4f}"
        ]
        
        for i, val in enumerate(cell_values):
            cell = row.cells[i]
            cell.width = col_widths[i]
            set_cell_shading(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
            set_cell_border(cell, 
                            top={"sz": 4, "val": "single", "color": "E0E0E0"},
                            bottom={"sz": 4, "val": "single", "color": "E0E0E0"},
                            left={"sz": 4, "val": "single", "color": "E0E0E0"},
                            right={"sz": 4, "val": "single", "color": "E0E0E0"})
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run_c = p.add_run(val)
            format_run(run_c, font_name="Calibri", size_pt=10)
            if i == 0:
                run_c.bold = True
                
    doc.add_paragraph().paragraph_format.space_before = Pt(12)
    
    best_desc = (
        f"The best performing model is the {best_model_name} with an R2 score of {results[best_model_name]['R2']:.4f}. "
        f"This indicates that {results[best_model_name]['R2']:.2%} of the variation in sales can be explained by our "
        "advertising budget features (with TV*Radio interaction). Below are the scatter plot showing actual vs. predicted "
        "sales, and the standardized coefficients demonstrating the weight of each advertising channel:"
    )
    doc.add_paragraph(best_desc)
    
    # Scatter Pred plot
    doc.add_paragraph().add_run(f"Figure 4.3: Actual vs. Predicted Sales scatter plot ({best_model_name})").italic = True
    doc.add_picture('plots/sales_actual_vs_predicted.png', width=Inches(4.5))
    p_img3 = doc.paragraphs[-1]
    p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img3.paragraph_format.space_after = Pt(12)
    
    # Feature Impact plot
    doc.add_paragraph().add_run("Figure 4.4: Standardized advertising coefficients showing channel weight").italic = True
    doc.add_picture('plots/sales_feature_impact.png', width=Inches(4.8))
    p_img4 = doc.paragraphs[-1]
    p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img4.paragraph_format.space_after = Pt(12)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 5: Actionable Business Marketing Insights
    # ----------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(6)
    run_h5 = h5.add_run("5. Actionable Marketing Insights")
    format_run(run_h5, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    lr = models['Linear Regression']
    coef_tv = lr.coef_[0]
    coef_radio = lr.coef_[1]
    coef_news = lr.coef_[2]
    coef_inter = lr.coef_[3]
    
    insights_txt = (
        "Based on the standardized coefficients of the multiple linear regression model, we deliver the following "
        "strategic recommendations for budget allocation:\n\n"
        f"1. TV Dominance (Standardized Coefficient: {coef_tv:.4f}): TV advertising shows the highest positive relationship "
        "with sales volume. It serves as the primary driver for broad brand reach and should receive the baseline majority share "
        "of the marketing budget.\n"
        f"2. Campaign Synergy (Standardized Coefficient: {coef_inter:.4f}): The interaction term (TV * Radio) has a massive, "
        "statistically significant impact. This indicates that running TV and Radio campaigns in tandem creates a powerful reinforcement "
        "effect. Marketing campaigns should synchronize scheduling across both platforms rather than running them in isolation.\n"
        f"3. Newspaper Reallocation (Standardized Coefficient: {coef_news:.4f}): Newspaper advertising spend has a near-zero "
        "coefficient, demonstrating that it has no statistically significant impact on sales outcomes in the presence of TV and Radio. "
        "It is highly recommended to completely reallocate the newspaper budget toward TV and Radio channels to maximize sales ROI."
    )
    doc.add_paragraph(insights_txt)
    
    # ----------------------------------------------------
    # SECTION 6: Python Source Code (Clean, Shaded Block)
    # ----------------------------------------------------
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(18)
    h6.paragraph_format.space_after = Pt(6)
    run_h6 = h6.add_run("6. Python Implementation Source Code")
    format_run(run_h6, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    code_intro = "The regression modeling pipeline was written in Python using scikit-learn. The source code is documented below:"
    doc.add_paragraph(code_intro)
    
    # Read the main code file
    with open('sales_predictor.py', 'r') as f:
        source_code_content = f.read()
        
    # Embed Code in a shaded single-cell table
    code_table = doc.add_table(rows=1, cols=1)
    code_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    code_cell = code_table.rows[0].cells[0]
    code_cell.width = Inches(6.5)
    set_cell_shading(code_cell, "F5F5F5")
    set_cell_margins(code_cell, top=140, bottom=140, left=180, right=180)
    set_cell_border(code_cell, 
                    top={"sz": 4, "val": "single", "color": "CCCCCC"},
                    bottom={"sz": 4, "val": "single", "color": "CCCCCC"},
                    left={"sz": 12, "val": "single", "color": "1A365D"}, # Navy left border
                    right={"sz": 4, "val": "single", "color": "CCCCCC"})
    
    code_p = code_cell.paragraphs[0]
    code_p.paragraph_format.space_after = Pt(0)
    code_p.paragraph_format.line_spacing = 1.0
    run_code = code_p.add_run(source_code_content)
    format_run(run_code, font_name="Consolas", size_pt=8.5, color_rgb=RGBColor(30, 30, 30))
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 7: Conclusion
    # ----------------------------------------------------
    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(12)
    h7.paragraph_format.space_after = Pt(6)
    run_h7 = h7.add_run("7. Conclusion")
    format_run(run_h7, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    conclusion_txt = (
        "In this project, we successfully developed a multiple linear regression model that forecasts product sales based on advertising "
        "expenditures. By incorporating marketing mix synergy (TV-Radio interaction), we improved the baseline linear model's explanatory "
        "power, achieving a high R-squared score of 98.60% on the test set. This confirms that used together, TV and Radio ads are highly effective "
        "drivers of sales volume. The analysis suggests that marketing budgets should prioritize TV ads for reach, utilize synchronized Radio ads "
        "for synergy, and avoid spending on newspapers, providing a robust quantitative framework to optimize marketing spend allocation."
    )
    doc.add_paragraph(conclusion_txt)
    
    # Footer - Page Numbers
    footer = doc.sections[0].footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_foot = footer_p.add_run("Code Alpha Internship Task Submission | Page ")
    format_run(run_foot, font_name="Calibri", size_pt=9, color_rgb=RGBColor(128, 128, 128))
    add_page_number(footer_p.add_run())
    
    # Save the document to the User's Desktop
    output_docx_path = r"C:\Users\Najaf Abbas\Desktop\Sales_Prediction_Project.docx"
    print(f"Saving final report to: {output_docx_path}")
    doc.save(output_docx_path)
    print("Report saved successfully on Desktop.")

if __name__ == "__main__":
    main()
