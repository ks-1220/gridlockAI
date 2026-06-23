"""
challan_generator.py — E-Challan Generation Utility
===================================================
Generates beautiful digital traffic ticket receipts in both HTML (for UI display
and instant printing) and PDF formats (using ReportLab).
"""

import os
from typing import Dict, Any, Optional
import cv2
import numpy as np
from datetime import datetime
from utils import ViolationRecord, Severity, VIOLATION_SEVERITY

# Fine structure based on severity of violation
FINE_STRUCTURE = {
    Severity.CRITICAL: 1000,
    Severity.MODERATE: 500,
    Severity.MINOR: 250
}

class ChallanGenerator:
    """Generates and manages E-Challan ticket outputs."""
    
    @staticmethod
    def calculate_fine(violation_type) -> int:
        """Calculate fine based on violation type severity."""
        severity = VIOLATION_SEVERITY.get(violation_type, Severity.MINOR)
        return FINE_STRUCTURE.get(severity, 250)

    @staticmethod
    def generate_html_challan(
        violation: ViolationRecord,
        vehicle_crop_b64: Optional[str] = None
    ) -> str:
        """
        Generates a premium HTML/CSS-styled Challan receipt.
        Suitable for displaying in Streamlit or printing directly.
        """
        fine_amount = ChallanGenerator.calculate_fine(violation.violation_type)
        severity_color = {
            Severity.CRITICAL: "#ef4444",
            Severity.MODERATE: "#f97316",
            Severity.MINOR: "#eab308"
        }.get(violation.severity, "#ffffff")

        # HTML template with modern dark glassmorphism styling
        html_template = f"""
        <div style="background: #1a1c23; border: 1px solid #2d3040; border-radius: 12px; padding: 24px; font-family: 'Inter', sans-serif; color: #e6e9ef; max-width: 600px; margin: 0 auto; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <!-- Header -->
            <div style="text-align: center; border-bottom: 2px solid #2d3040; padding-bottom: 16px; margin-bottom: 20px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #8b8fa3; margin-bottom: 4px;">Ministry of Road Transport & Highways</div>
                <div style="font-size: 20px; font-weight: 800; color: #4f8ff7; letter-spacing: 0.5px;">E-CHALLAN INFRINGEMENT TICKET</div>
                <div style="font-size: 11px; font-weight: 500; color: #8b8fa3; margin-top: 4px;">AUTOMATED SURVEILLANCE SYSTEM (GRIDLOCK-AI)</div>
            </div>

            <!-- Meta details grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; font-size: 13px;">
                <div>
                    <span style="color: #8b8fa3; display: block; margin-bottom: 2px; text-transform: uppercase; font-size: 10px; font-weight: 700;">Challan Number</span>
                    <strong style="color: #e6e9ef;">{violation.violation_id}</strong>
                </div>
                <div>
                    <span style="color: #8b8fa3; display: block; margin-bottom: 2px; text-transform: uppercase; font-size: 10px; font-weight: 700;">Date & Time</span>
                    <strong style="color: #e6e9ef;">{violation.timestamp}</strong>
                </div>
                <div>
                    <span style="color: #8b8fa3; display: block; margin-bottom: 2px; text-transform: uppercase; font-size: 10px; font-weight: 700;">Vehicle Reference</span>
                    <strong style="color: #e6e9ef;">{violation.vehicle_id}</strong>
                </div>
                <div>
                    <span style="color: #8b8fa3; display: block; margin-bottom: 2px; text-transform: uppercase; font-size: 10px; font-weight: 700;">Extracted License Plate</span>
                    <strong style="color: #4f8ff7; font-size: 14px;">{violation.license_plate or "PLATE UNREADABLE (PENDING MANUAL AUDIT)"}</strong>
                </div>
            </div>

            <!-- Infraction Detail Section -->
            <div style="background: #21232d; border-left: 4px solid {severity_color}; border-radius: 6px; padding: 16px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #8b8fa3; font-size: 11px; text-transform: uppercase; font-weight: 700;">Infraction Detected</span>
                        <div style="font-size: 16px; font-weight: 700; color: #e6e9ef; margin-top: 2px;">{violation.violation_type.value}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="display: inline-block; background: {severity_color}22; color: {severity_color}; border: 1px solid {severity_color}; font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 4px; text-transform: uppercase; margin-bottom: 4px;">{violation.severity.value}</span>
                        <div style="font-size: 12px; color: #8b8fa3;">System Confidence: <strong>{violation.confidence * 100:.1f}%</strong></div>
                    </div>
                </div>
            </div>
        """

        if vehicle_crop_b64:
            html_template += f"""
            <!-- Bounding Box Crop Visual -->
            <div style="margin-bottom: 24px; text-align: center;">
                <span style="color: #8b8fa3; display: block; margin-bottom: 6px; text-transform: uppercase; font-size: 10px; font-weight: 700; text-align: left;">Evidence Bounding Box Crop</span>
                <img src="data:image/jpeg;base64,{vehicle_crop_b64}" style="max-width: 100%; border: 1px solid #2d3040; border-radius: 6px; max-height: 180px; object-fit: contain; background: #0f1117;" />
            </div>
            """

        html_template += f"""
            <!-- Summary Billing & Legal Notice -->
            <div style="border-top: 1px dashed #2d3040; padding-top: 16px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 16px; font-weight: 800;">
                    <span>Penalty Assessment:</span>
                    <span style="color: #ef4444; font-size: 20px;">₹ {fine_amount:,.2f}</span>
                </div>
                <p style="font-size: 10px; color: #8b8fa3; line-height: 1.4; margin-top: 12px; text-align: justify;">
                    <strong>LEGAL NOTICE:</strong> This document serves as an official notice of a traffic regulation violation captured by automatic camera feed analysis. Please settle this fine within 15 days of issuance via the official portal or submit an appeal. Unpaid penalties may lead to registration suspension.
                </p>
            </div>

            <!-- Footer Pay QR -->
            <div style="display: flex; align-items: center; background: #14151b; border-radius: 8px; padding: 12px;">
                <div style="flex-grow: 1; font-size: 11px; color: #8b8fa3; line-height: 1.3;">
                    <strong>Scan Mock QR Code to Pay:</strong><br/>
                    Instantly clear pending traffic tickets using authorized digital wallet integrations.
                </div>
                <div style="background: white; padding: 4px; border-radius: 4px; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin-left: 12px;">
                    <!-- Simple Mock QR Representation using SVG -->
                    <svg width="50" height="50" viewBox="0 0 100 100" style="display: block;">
                        <rect x="0" y="0" width="30" height="30" fill="black"/>
                        <rect x="5" y="5" width="20" height="20" fill="white"/>
                        <rect x="10" y="10" width="10" height="10" fill="black"/>
                        <rect x="70" y="0" width="30" height="30" fill="black"/>
                        <rect x="75" y="5" width="20" height="20" fill="white"/>
                        <rect x="80" y="10" width="10" height="10" fill="black"/>
                        <rect x="0" y="70" width="30" height="30" fill="black"/>
                        <rect x="5" y="75" width="20" height="20" fill="white"/>
                        <rect x="10" y="80" width="10" height="10" fill="black"/>
                        <rect x="40" y="40" width="20" height="20" fill="black"/>
                        <rect x="80" y="80" width="20" height="20" fill="black"/>
                        <rect x="50" y="80" width="10" height="10" fill="black"/>
                        <rect x="80" y="50" width="10" height="10" fill="black"/>
                        <rect x="45" y="0" width="10" height="20" fill="black"/>
                    </svg>
                </div>
            </div>
        </div>
        """
        return html_template

    @staticmethod
    def generate_pdf_challan(
        violation: ViolationRecord,
        vehicle_crop: Optional[np.ndarray] = None
    ) -> Optional[bytes]:
        """
        Generates a professional PDF Challan invoice using reportlab.
        Falls back to HTML or returns None if reportlab is unavailable.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import io
            
            fine_amount = ChallanGenerator.calculate_fine(violation.violation_type)
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            story = []

            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=18,
                leading=22,
                textColor=colors.HexColor('#1f3a60'),
                alignment=1, # Centered
                spaceAfter=5
            )
            
            subtitle_style = ParagraphStyle(
                'SubTitleStyle',
                parent=styles['Normal'],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor('#6b7280'),
                alignment=1,
                spaceAfter=15
            )

            section_title = ParagraphStyle(
                'SectionTitle',
                parent=styles['Heading2'],
                fontSize=11,
                leading=14,
                textColor=colors.HexColor('#374151'),
                spaceBefore=10,
                spaceAfter=6,
                borderColor=colors.HexColor('#e5e7eb'),
                borderWidth=1,
                borderPadding=4
            )

            normal_style = ParagraphStyle(
                'NormalStyle',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor('#111827')
            )
            
            alert_style = ParagraphStyle(
                'AlertStyle',
                parent=styles['Normal'],
                fontSize=11,
                leading=15,
                textColor=colors.HexColor('#b91c1c'),
                fontName='Helvetica-Bold'
            )

            # Header
            story.append(Paragraph("MINISTRY OF ROAD TRANSPORT & HIGHWAYS", ParagraphStyle('Gov', fontSize=8, alignment=1, textColor=colors.HexColor('#4b5563'))))
            story.append(Paragraph("E-CHALLAN TRAFFIC INFRINGEMENT RECEIPT", title_style))
            story.append(Paragraph("AUTOMATED ROAD SAFETY SURVEILLANCE ENGINE (GRIDLOCK-AI)", subtitle_style))
            
            # Metadata Table
            meta_data = [
                [Paragraph("<b>Challan Number:</b>", normal_style), Paragraph(violation.violation_id, normal_style),
                 Paragraph("<b>Date & Time:</b>", normal_style), Paragraph(violation.timestamp, normal_style)],
                [Paragraph("<b>Vehicle ID:</b>", normal_style), Paragraph(violation.vehicle_id, normal_style),
                 Paragraph("<b>License Plate:</b>", normal_style), Paragraph(violation.license_plate or "N/A (UNREADABLE)", normal_style)]
            ]
            t_meta = Table(meta_data, colWidths=[1.5*inch, 2.0*inch, 1.25*inch, 2.25*inch])
            t_meta.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_meta)
            story.append(Spacer(1, 15))

            # Infraction Assessment Table
            severity_col = colors.HexColor('#ef4444') if violation.severity == Severity.CRITICAL else colors.HexColor('#f97316')
            infraction_data = [
                [Paragraph("<b>Violation Description</b>", normal_style), Paragraph("<b>Severity</b>", normal_style), Paragraph("<b>Penalty Amount</b>", normal_style)],
                [Paragraph(violation.violation_type.value, normal_style), 
                 Paragraph(f"<font color='{severity_col.hexval()}'><b>{violation.severity.value}</b></font>", normal_style), 
                 Paragraph(f"<b>INR {fine_amount:,.2f}</b>", alert_style)]
            ]
            t_infraction = Table(infraction_data, colWidths=[3.5*inch, 1.5*inch, 2.0*inch])
            t_infraction.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(t_infraction)
            story.append(Spacer(1, 15))

            # Evidence Image Box
            if vehicle_crop is not None:
                # Save crop temporarily for ReportLab
                temp_filename = f"temp_challan_crop_{violation.violation_id}.jpg"
                cv2.imwrite(temp_filename, vehicle_crop)
                try:
                    # Resize proportionally for layout constraints
                    img_w, img_h = 4.0 * inch, 2.0 * inch
                    rl_img = RLImage(temp_filename, width=img_w, height=img_h)
                    rl_img.hAlign = 'CENTER'
                    story.append(Paragraph("<b>PHOTOGRAPHIC INFRACTION EVIDENCE</b>", section_title))
                    story.append(rl_img)
                    story.append(Spacer(1, 10))
                finally:
                    # Clean up temp file immediately after placing in story list
                    if os.path.exists(temp_filename):
                        # We delete it later in buffer closing, but we can register a trigger
                        pass

            # Legal Notice
            story.append(Paragraph("<b>LEGAL COMPLIANCE NOTICE:</b>", section_title))
            legal_text = (
                "This receipt constitutes an officially validated notice of traffic regulation infringement. "
                "The infraction was registered using an automated deep learning pipeline (RT-DETR + Zero-shot CLIP + OCR). "
                "Please pay this penalty within 15 calendar days from the timestamp listed above. "
                "Appeals or disputes can be filed at the standard highway authority regional offices. "
                "Failure to complete payment within the deadline may result in a formal police record and suspension of the registration."
            )
            story.append(Paragraph(legal_text, ParagraphStyle('Legal', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor('#4b5563'))))
            story.append(Spacer(1, 20))

            # Signatures mock
            sig_data = [
                [Paragraph("<b>Verified By:</b> Gridlock AI Automated Agent", normal_style), 
                 Paragraph("<b>Payment Status:</b> PENDING", alert_style)]
            ]
            t_sig = Table(sig_data, colWidths=[4.0*inch, 3.0*inch])
            t_sig.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor('#d1d5db')),
                ('TOPPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(t_sig)

            # Build Document
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()

            # Clean up temp file
            temp_filename = f"temp_challan_crop_{violation.violation_id}.jpg"
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

            return pdf_data

        except Exception as e:
            # Fallback if ReportLab fails or is not installed
            print(f"Reportlab failed: {e}")
            return None
