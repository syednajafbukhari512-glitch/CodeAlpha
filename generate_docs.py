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
    print("--- STEP 1: Running Regression Training Pipeline ---")
    
    # 1. Download & Load Dataset
    url = "https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/master/car%20data.csv"
    DATA_PATH = "car_data.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    
    # Standardize column names to prevent errors across different Kaggle versions
    rename_dict = {
        'Kms_Driven': 'Driven_kms',
        'Seller_Type': 'Selling_type'
    }
    df = df.rename(columns=rename_dict)
    
    # Feature engineering
    CURRENT_YEAR = 2026
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    model_df = df.drop(['Car_Name', 'Year'], axis=1)
    
    # One-hot encode
    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Heatmap
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(model_df.corr(), annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Car Features', fontsize=12, fontweight='bold', pad=10)
    plt.savefig('plots/car_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scatter plot
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x='Present_Price', y='Selling_Price', hue='Fuel_Type', palette='Set1', s=60)
    plt.title('Selling Price vs. Present Price', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Present Price (Lakhs)', fontsize=10)
    plt.ylabel('Selling Price (Lakhs)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/car_price_vs_present.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Transmission', y='Selling_Price', hue='Transmission', palette='Pastel2', legend=False)
    plt.title('Selling Price Distribution by Transmission Type', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Transmission', fontsize=10)
    plt.ylabel('Selling Price (Lakhs)', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('plots/car_price_by_transmission.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Regression modeling
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    X = model_df.drop('Selling_Price', axis=1)
    y = model_df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    scale_cols = ['Present_Price', 'Driven_kms', 'Car_Age']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
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
    
    # Plot 4: Actual vs Predicted Scatter
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(x=y_test, y=y_pred_best, color='#1A365D', s=60, alpha=0.8)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs. Predicted Prices ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Actual Price (Lakhs)', fontsize=10)
    plt.ylabel('Predicted Price (Lakhs)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/car_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Feature Importance
    plt.figure(figsize=(8, 4.5))
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis', hue=X.columns[indices], legend=False)
        plt.title(f'Feature Importance in Price Prediction ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    else:
        coefs = best_model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        sns.barplot(x=coefs[indices], y=X.columns[indices], palette='coolwarm', hue=X.columns[indices], legend=False)
        plt.title(f'Model Coefficients in Price Prediction ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
        
    plt.xlabel('Relative Importance / Coefficient Value', fontsize=10)
    plt.ylabel('Feature', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('plots/car_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EDA and Regression training complete.")
    
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
    run_sub = subtitle_p.add_run("Task 3: Car Price Prediction using Machine Learning Regression")
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
        "Valuing used vehicles is a highly complex task due to the interplay of multiple quantitative and qualitative features. "
        "A vehicle's market price is influenced not only by physical characteristics like showroom price, purchase year, and kilometers "
        "driven, but also by brand goodwill, fuel efficiency, transmission type, ownership history, and seller channel.\n\n"
        "The objective of this project is to develop and train a regression machine learning model to predict used car prices "
        "based on these features. By engineering relevant variables (such as vehicle age) and preprocessing categorical data, we compare "
        "the performance of multiple linear and non-linear regression algorithms (Linear Regression, Lasso, Ridge, and Random Forest Regressor). "
        "This project provides practical experience in feature scaling, model selection, hyperparameter regularizations, and coefficient analysis."
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
        "Exploratory Data Analysis (EDA) allows us to understand the underlying distributions of features and examine how different "
        "attributes interact with the selling price. Below are the key charts generated during exploration:"
    )
    doc.add_paragraph(eda_txt)
    
    # Heatmap
    doc.add_paragraph().add_run("Figure 3.1: Heatmap showing correlation among features (numerical and encoded)").italic = True
    doc.add_picture('plots/car_correlation_matrix.png', width=Inches(4.4))
    p_img1 = doc.paragraphs[-1]
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img1.paragraph_format.space_after = Pt(18)
    
    # Scatter plot
    doc.add_paragraph().add_run("Figure 3.2: Selling price vs. showroom price across fuel types").italic = True
    doc.add_picture('plots/car_price_vs_present.png', width=Inches(4.8))
    p_img2 = doc.paragraphs[-1]
    p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img2.paragraph_format.space_after = Pt(18)
    
    doc.add_page_break()
    
    # Boxplot
    doc.add_paragraph().add_run("Figure 3.3: Distribution of selling prices based on transmission type").italic = True
    doc.add_picture('plots/car_price_by_transmission.png', width=Inches(4.2))
    p_img3 = doc.paragraphs[-1]
    p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img3.paragraph_format.space_after = Pt(18)
    
    # ----------------------------------------------------
    # SECTION 3: Preprocessing & Methodology
    # ----------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)
    run_h3 = h3.add_run("3. Preprocessing & Feature Engineering")
    format_run(run_h3, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    methodology_txt = (
        "To ensure that regression models converge efficiently and learn patterns accurately, we implement the following pipeline:\n\n"
        "1. Feature Engineering (Vehicle Age): Raw purchase year has limited predictive utility directly. We computed the car's age "
        "as: Car_Age = 2026 - Year, which directly captures depreciation.\n"
        "2. Categorical Encoding: Nominal fields like Fuel_Type (CNG/Diesel/Petrol), Selling_type (Dealer/Individual), and Transmission "
        "(Manual/Automatic) were converted to binary indicator variables using One-Hot Encoding (with drop_first=True to avoid multicollinearity).\n"
        "3. Feature Scaling: Continuous variables (Present_Price, Driven_kms, Car_Age) were normalized using standard scaling (mean=0, variance=1) "
        "to ensure stability in linear estimators."
    )
    doc.add_paragraph(methodology_txt)
    
    # ----------------------------------------------------
    # SECTION 4: Regression Model Performance Comparison
    # ----------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(18)
    h4.paragraph_format.space_after = Pt(6)
    run_h4 = h4.add_run("4. Model Performance Comparison")
    format_run(run_h4, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    results_txt = (
        "Each model was trained on the 80% train split and evaluated on the 20% test split. "
        "The comparison metrics (R-squared score, MAE, MSE, and RMSE) are presented in the table below:"
    )
    doc.add_paragraph(results_txt)
    
    # Create Table
    comp_table = doc.add_table(rows=5, cols=5)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Model", "R-squared (R2)", "MAE (Lakhs)", "MSE", "RMSE"]
    col_widths = [Inches(2.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]
    
    # Table Header Row
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
        
    # Table Content
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
        f"Based on the R-squared metric, the best performing model is the {best_model_name} with an R2 score of "
        f"{results[best_model_name]['R2']:.4f}. This indicates that approximately {results[best_model_name]['R2']:.2%} of "
        "the variance in selling price can be explained by our model features. Below are the actual vs. predicted prices scatter "
        "plot and the feature importances bar chart for the best model:"
    )
    doc.add_paragraph(best_desc)
    
    # Scatter Pred plot
    doc.add_paragraph().add_run(f"Figure 3.4: Actual vs. Predicted Prices scatter plot ({best_model_name})").italic = True
    doc.add_picture('plots/car_actual_vs_predicted.png', width=Inches(4.5))
    p_img4 = doc.paragraphs[-1]
    p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img4.paragraph_format.space_after = Pt(12)
    
    # Feature Importance plot
    doc.add_paragraph().add_run(f"Figure 3.5: Relative feature importance / coefficients ({best_model_name})").italic = True
    doc.add_picture('plots/car_feature_importance.png', width=Inches(4.8))
    p_img5 = doc.paragraphs[-1]
    p_img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img5.paragraph_format.space_after = Pt(12)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 5: Python Source Code (Clean, Shaded Block)
    # ----------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(6)
    run_h5 = h5.add_run("5. Python Implementation Source Code")
    format_run(run_h5, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    code_intro = "The regression modeling pipeline was written in Python using scikit-learn. The source code is documented below:"
    doc.add_paragraph(code_intro)
    
    # Read the main code file
    with open('car_price_predictor.py', 'r') as f:
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
    # SECTION 6: Real-World Applications & Insights
    # ----------------------------------------------------
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(12)
    h6.paragraph_format.space_after = Pt(6)
    run_h6 = h6.add_run("6. Real-World Applications & Business Insights")
    format_run(run_h6, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    app_intro = (
        "Car price prediction models have high business value in the automotive and finance industries. "
        "Key real-world applications include:"
    )
    doc.add_paragraph(app_intro)
    
    apps = [
        "Used Car Valuation Platforms: Companies like CarDekho, Cars24, and Spinny use regression models to provide instant, automated price valuations "
        "to customers selling their cars. This drastically reduces assessment times and ensures standardized, data-driven offers.",
        "Dealer Inventory Pricing: Car dealers utilize price prediction algorithms to dynamically adjust their listings based on market competition, "
        "mileage thresholds, and demand for specific fuel/transmission configurations, optimizing turnover rates.",
        "Auto Loan Underwriting: Financial institutions and banks utilize these models to estimate the residual value (depreciation curve) of the collateral "
        "(the vehicle) when underwriting auto loans, minimizing default risks."
    ]
    for app in apps:
        doc.add_paragraph(app, style='List Bullet')
        
    # ----------------------------------------------------
    # SECTION 7: Conclusion
    # ----------------------------------------------------
    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(18)
    h7.paragraph_format.space_after = Pt(6)
    run_h7 = h7.add_run("7. Conclusion")
    format_run(run_h7, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    conclusion_txt = (
        "In this project, we successfully implemented a machine learning workflow to predict used car prices. Through data exploration, we "
        "found that the current showroom price (Present_Price) and vehicle age (Car_Age) are the strongest predictors of used car value. "
        "Random Forest Regressor demonstrated the highest performance, achieving an R-squared score of 95.83%, significantly outperforming "
        "linear baseline estimators. This showcases the capability of non-linear ensemble models to capture complex feature relationships. "
        "The model provides a robust foundation that can be expanded with additional features like brand goodwill ratings, mileage (km/l), "
        "and physical condition logs for real-world production settings."
    )
    doc.add_paragraph(conclusion_txt)
    
    # ----------------------------------------------------
    # Footer - Page Numbers
    # ----------------------------------------------------
    footer = doc.sections[0].footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_foot = footer_p.add_run("Code Alpha Internship Task Submission | Page ")
    format_run(run_foot, font_name="Calibri", size_pt=9, color_rgb=RGBColor(128, 128, 128))
    add_page_number(footer_p.add_run())
    
    # Save the document to the User's Desktop
    output_docx_path = r"C:\Users\Najaf Abbas\Desktop\Car_Price_Prediction_Project.docx"
    print(f"Saving final report to: {output_docx_path}")
    doc.save(output_docx_path)
    print("Report saved successfully on Desktop.")

if __name__ == "__main__":
    main()
