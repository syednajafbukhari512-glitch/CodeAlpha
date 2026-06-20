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
    print("--- STEP 1: Running Data Analysis & EDA Pipeline ---")
    
    # 1. Download & Load Dataset
    urls = [
        "https://raw.githubusercontent.com/proteendas/OIBSIP-DS-03/main/Unemployment_Rate_upto_11_2020.csv",
        "https://raw.githubusercontent.com/Tosa9/CodeAlpha_UnemploymentAnalysis/main/data/Unemployment_Rate_upto_11_2020.csv"
    ]
    DATA_PATH = "Unemployment_Rate_upto_11_2020.csv"
    downloaded = False
    
    if not os.path.exists(DATA_PATH):
        for url in urls:
            print(f"Trying download from: {url}")
            try:
                urllib.request.urlretrieve(url, DATA_PATH)
                print("Download successful.")
                downloaded = True
                break
            except Exception:
                pass
        if not downloaded:
            print("Download failed.")
            return
            
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Region', 'Date', 'Estimated Unemployment Rate (%)'])
    df['Date'] = df['Date'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Monthly Trend
    monthly_trend = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(data=monthly_trend, x='Date', y='Estimated Unemployment Rate (%)', marker='o', linewidth=2.5, color='#1A365D')
    plt.title('Average Monthly Unemployment Rate in India (2020)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Unemployment Rate (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvspan('2020-04-01', '2020-06-30', color='red', alpha=0.1, label='Lockdown Peak Period')
    plt.legend(fontsize=9)
    plt.savefig('plots/unemployment_monthly_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Region Trend
    region_data = df.groupby('Region.1')['Estimated Unemployment Rate (%)'].mean().reset_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=region_data.sort_values(by='Estimated Unemployment Rate (%)', ascending=False), 
                x='Region.1', y='Estimated Unemployment Rate (%)', hue='Region.1', palette='Blues_r', legend=False)
    plt.title('Average Unemployment Rate by Geographical Region', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Region', fontsize=10)
    plt.ylabel('Average Unemployment Rate (%)', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('plots/unemployment_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Top States
    state_data = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
    top_states = state_data.sort_values(by='Estimated Unemployment Rate (%)', ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_states, x='Estimated Unemployment Rate (%)', y='Region', hue='Region', palette='Reds_r', legend=False)
    plt.title('Top 12 Indian States/UTs with Highest Average Unemployment (2020)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Average Unemployment Rate (%)', fontsize=10)
    plt.ylabel('State / UT', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('plots/unemployment_top_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # COVID-19 Phase Analysis
    def get_phase(date):
        if date < pd.Timestamp('2020-04-01'):
            return 'Pre-Lockdown (Jan-Mar)'
        elif date <= pd.Timestamp('2020-06-30'):
            return 'Lockdown Peak (Apr-Jun)'
        else:
            return 'Post-Lockdown (Jul-Nov)'
            
    df['Lockdown_Phase'] = df['Date'].apply(get_phase)
    phase_analysis = df.groupby('Lockdown_Phase')[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].mean().reset_index()
    
    # Plot 4: Boxplot showing distributions of unemployment rates during phases
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x='Lockdown_Phase', y='Estimated Unemployment Rate (%)', hue='Lockdown_Phase', palette='Set2', legend=False)
    plt.title('Unemployment Rate Distribution across COVID-19 Phases', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Lockdown Phase', fontsize=10)
    plt.ylabel('Unemployment Rate (%)', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig('plots/unemployment_phase_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EDA and analysis plots generated.")
    
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
    font.color.rgb = RGBColor(51, 51, 51) # Charcoal
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)
    
    # ----------------------------------------------------
    # COVER / SUBMISSION HEADER (Clean, Professional style)
    # ----------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(12)
    run_title = title_p.add_run("Code Alpha Internship Task Submission")
    format_run(run_title, font_name="Calibri Light", size_pt=26, bold=True, color_rgb=RGBColor(26, 54, 93)) # Deep Navy
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(36)
    run_sub = subtitle_p.add_run("Task 2: Unemployment Analysis in India with Python")
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
        # Format label cell
        cell_lbl = meta_table.rows[idx].cells[0]
        cell_lbl.width = Inches(2.0)
        p_lbl = cell_lbl.paragraphs[0]
        run_lbl = p_lbl.add_run(label)
        format_run(run_lbl, font_name="Calibri", size_pt=11, bold=True, color_rgb=RGBColor(26, 54, 93))
        
        # Format value cell
        cell_val = meta_table.rows[idx].cells[1]
        cell_val.width = Inches(4.0)
        p_val = cell_val.paragraphs[0]
        run_val = p_val.add_run(val)
        format_run(run_val, font_name="Calibri", size_pt=11)
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 1: Introduction & Project Objectives
    # ----------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    run_h1 = h1.add_run("1. Introduction & Objectives")
    format_run(run_h1, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    intro_txt = (
        "Unemployment is a key macroeconomic indicator representing the percentage of the eligible labour force that is actively "
        "seeking work but remains unemployed. The COVID-19 pandemic of 2020 created unprecedented challenges for economic stability, "
        "triggering nationwide lockdowns that severely impacted businesses, supply chains, and employment opportunities.\n\n"
        "This project focuses on analyzing the unemployment rate across different states and union territories in India "
        "during the year 2020. Using data collected by the Center for Monitoring Indian Economy (CMIE), we perform data cleaning, "
        "exploratory analysis, and trend visualization. The goal is to evaluate the direct impact of Covid-19 lock-downs on the "
        "workforce, trace regional differences, identify seasonal trends, and present economic policy insights."
    )
    doc.add_paragraph(intro_txt)
    
    # ----------------------------------------------------
    # SECTION 2: Exploratory Data Analysis & Trends
    # ----------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)
    run_h2 = h2.add_run("2. Exploratory Data Analysis & Visualizations")
    format_run(run_h2, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    eda_txt = (
        "A temporal analysis of the unemployment rate shows a dramatic spike during the months of April and May 2020, coinciding "
        "with the implementation of India's strict nationwide lockdown. Below are the key visualizations summarizing the trends:"
    )
    doc.add_paragraph(eda_txt)
    
    # Line plot
    doc.add_paragraph().add_run("Figure 2.1: Monthly average unemployment rate trend across India (2020)").italic = True
    doc.add_picture('plots/unemployment_monthly_trend.png', width=Inches(5.0))
    p_img1 = doc.paragraphs[-1]
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img1.paragraph_format.space_after = Pt(18)
    
    # Region plot
    doc.add_paragraph().add_run("Figure 2.2: Geographical region-wise average unemployment rate comparison").italic = True
    doc.add_picture('plots/unemployment_by_region.png', width=Inches(4.2))
    p_img2 = doc.paragraphs[-1]
    p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img2.paragraph_format.space_after = Pt(18)
    
    doc.add_page_break()
    
    # Top States plot
    doc.add_paragraph().add_run("Figure 2.3: Top Indian States and UTs with the highest average unemployment rates").italic = True
    doc.add_picture('plots/unemployment_top_states.png', width=Inches(5.0))
    p_img3 = doc.paragraphs[-1]
    p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img3.paragraph_format.space_after = Pt(18)
    
    # ----------------------------------------------------
    # SECTION 3: COVID-19 Impact Assessment
    # ----------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)
    run_h3 = h3.add_run("3. COVID-19 Impact Assessment (Phase Analysis)")
    format_run(run_h3, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    phase_txt = (
        "To thoroughly investigate the impact of COVID-19, we partitioned the dataset into three distinct timeframes:\n"
        "1. Pre-Lockdown (January - March 2020): Normal economic activity prior to the outbreak.\n"
        "2. Lockdown Peak (April - June 2020): Period of severe containment measures and commercial shutdown.\n"
        "3. Post-Lockdown Recovery (July - November 2020): Phase of partial economic reopening and gradual normalization.\n\n"
        "The calculated average indicators for each phase are summarized below:"
    )
    doc.add_paragraph(phase_txt)
    
    # Create Table of phase statistics
    stat_table = doc.add_table(rows=4, cols=4)
    stat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Phase", "Unemployment Rate (%)", "Employed Workforce (Mean)", "Labour Participation (%)"]
    col_widths = [Inches(2.5), Inches(1.3), Inches(1.5), Inches(1.2)]
    
    # Table Header Row
    hdr_row = stat_table.rows[0]
    for i, title in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = col_widths[i]
        set_cell_shading(cell, "1A365D") # Navy blue
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run_h = p.add_run(title)
        format_run(run_h, font_name="Calibri", size_pt=10, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    # Table Content
    for idx, row_data in phase_analysis.iterrows():
        row = stat_table.rows[idx + 1]
        bg_color = "F9F9F9" if idx % 2 == 0 else "FFFFFF"
        
        phase_lbl = row_data['Lockdown_Phase']
        unemp_val = f"{row_data['Estimated Unemployment Rate (%)']:.2f}%"
        emp_val = f"{row_data['Estimated Employed']:,.0f}"
        part_val = f"{row_data['Estimated Labour Participation Rate (%)']:.2f}%"
        
        cell_vals = [phase_lbl, unemp_val, emp_val, part_val]
        
        for i, val in enumerate(cell_vals):
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
    
    # Phase boxplot image
    doc.add_paragraph().add_run("Figure 2.4: Unemployment rate distribution across COVID-19 phases").italic = True
    doc.add_picture('plots/unemployment_phase_boxplot.png', width=Inches(4.5))
    p_img4 = doc.paragraphs[-1]
    p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img4.paragraph_format.space_after = Pt(12)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 4: Python Source Code (Clean, Shaded Block)
    # ----------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(6)
    run_h4 = h4.add_run("4. Python Implementation Source Code")
    format_run(run_h4, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    code_intro = "The entire data cleaning, processing, and analysis workflow was implemented in Python. The source code is documented below:"
    doc.add_paragraph(code_intro)
    
    # Read the main code file
    with open('unemployment_analyser.py', 'r') as f:
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
    # SECTION 5: Policy Recommendations & Insights
    # ----------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(6)
    run_h5 = h5.add_run("5. Economic & Social Policy Recommendations")
    format_run(run_h5, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    policy_intro = (
        "Based on the trends observed during the COVID-19 pandemic phases, the following key insights and policy "
        "recommendations are presented to mitigate future economic shocks and support labor market recovery:"
    )
    doc.add_paragraph(policy_intro)
    
    policies = [
        "Strengthening Social Safety Nets: During economic shutdowns, social safety net programs (such as MGNREGA in rural India) "
        "must be temporarily expanded. Providing immediate wage guarantees helps sustain purchasing power and prevents deep poverty spikes.",
        "State-specific Targeted Aid: The data demonstrates that states like Haryana, Tripura, and Jharkhand experienced disproportionately "
        "high average unemployment rates (exceeding 20%). Future relief packages, industrial credits, and development funds should be "
        "allocated dynamically based on real-time state-level unemployment metrics rather than flat nationwide distributions.",
        "Promotion of Flexible and Remote Work Infrastructures: The service sector and urban areas showed distinct fluctuations. Policymakers "
        "should incentivize digital infrastructure and formulate frameworks for remote working, making businesses more resilient to physical containment measures.",
        "Reskilling and Skill-Mapping Programs: Displaced workers require rapid reskilling to align with post-lockdown growth sectors (such as e-commerce, logistics, and digital services). Establishing state-funded online platforms for vocational training can facilitate rapid labour reallocation."
    ]
    for pol in policies:
        doc.add_paragraph(pol, style='List Bullet')
        
    # ----------------------------------------------------
    # SECTION 6: Conclusion
    # ----------------------------------------------------
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(18)
    h6.paragraph_format.space_after = Pt(6)
    run_h6 = h6.add_run("6. Conclusion")
    format_run(run_h6, font_name="Calibri Light", size_pt=18, bold=True, color_rgb=RGBColor(26, 54, 93))
    
    conclusion_txt = (
        "This unemployment analysis illustrates the profound impact of the COVID-19 pandemic on India's workforce. The average unemployment "
        "rate spiked from 9.47% during the pre-lockdown months to a historical high of 19.34% during the peak of the lockdown. While the "
        "recovery phase saw a decline to 10.37%, the labor market remained strained, indicating structural damage. Regional disparities "
        "were highly evident, with northern and eastern regions bearing the brunt of the shock. These insights emphasize the need for "
        "resilient, adaptive economic policies, decentralized crisis management, and robust social welfare structures to support "
        "the workforce through major macro-economic crises."
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
    output_docx_path = r"C:\Users\Najaf Abbas\Desktop\Unemployment_Analysis_Report.docx"
    print(f"Saving final report to: {output_docx_path}")
    doc.save(output_docx_path)
    print("Report saved successfully on Desktop.")

if __name__ == "__main__":
    main()
