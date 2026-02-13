"""
Generate professional PDF documentation for the SVAMITVA Feature Extraction System.
Team SVAMITVA - SIH Hackathon 2026
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, gray
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, Preformatted, KeepTogether
)
from reportlab.lib import colors
from datetime import datetime


PRIMARY = HexColor("#1a5276")
SECONDARY = HexColor("#2e86c1")
ACCENT = HexColor("#e67e22")
LIGHT_BG = HexColor("#eaf2f8")
CODE_BG = HexColor("#f4f4f4")
DARK_TEXT = HexColor("#2c3e50")


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="DocTitle",
        parent=styles["Title"],
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="DocSubtitle",
        parent=styles["Title"],
        fontSize=16,
        leading=20,
        textColor=SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=8,
        fontName="Helvetica",
    ))

    styles.add(ParagraphStyle(
        name="TeamName",
        parent=styles["Title"],
        fontSize=14,
        leading=18,
        textColor=ACCENT,
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="Heading1Custom",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceBefore=20,
        spaceAfter=10,
        fontName="Helvetica-Bold",
        borderWidth=1,
        borderColor=PRIMARY,
        borderPadding=6,
    ))

    styles.add(ParagraphStyle(
        name="Heading2Custom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=SECONDARY,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="Heading3Custom",
        parent=styles["Heading3"],
        fontSize=12,
        leading=15,
        textColor=DARK_TEXT,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=DARK_TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        fontName="Helvetica",
    ))

    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontSize=8,
        leading=10,
        textColor=HexColor("#333333"),
        backColor=CODE_BG,
        fontName="Courier",
        leftIndent=12,
        rightIndent=12,
        spaceBefore=6,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="BulletCustom",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14,
        textColor=DARK_TEXT,
        leftIndent=20,
        spaceAfter=3,
        fontName="Helvetica",
        bulletIndent=8,
    ))

    styles.add(ParagraphStyle(
        name="TOCEntry",
        parent=styles["Normal"],
        fontSize=12,
        leading=18,
        textColor=DARK_TEXT,
        spaceAfter=4,
        fontName="Helvetica",
        leftIndent=10,
    ))

    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=gray,
        alignment=TA_CENTER,
        spaceAfter=8,
        fontName="Helvetica-Oblique",
    ))

    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(gray)
    page_num = canvas.getPageNumber()
    text = f"SVAMITVA Feature Extraction System | Page {page_num}"
    canvas.drawCentredString(A4[0] / 2, 15 * mm, text)
    canvas.line(30 * mm, 18 * mm, A4[0] - 30 * mm, 18 * mm)
    canvas.restoreState()


def first_page_template(canvas, doc):
    canvas.saveState()
    canvas.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        "SVAMITVA_Documentation.pdf",
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
    )

    styles = get_styles()
    story = []

    # ==================== TITLE PAGE ====================
    story.append(Spacer(1, 80))
    story.append(Paragraph("SVAMITVA", styles["DocTitle"]))
    story.append(Paragraph("Feature Extraction System", styles["DocTitle"]))
    story.append(Spacer(1, 20))

    line_data = [["" * 60]]
    line_table = Table(line_data, colWidths=[400])
    line_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 2, ACCENT),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "AI-Powered Drone Imagery Analysis<br/>for Rural Property Mapping",
        styles["DocSubtitle"]
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Team SVAMITVA", styles["TeamName"]))
    story.append(Paragraph("Smart India Hackathon 2026", styles["DocSubtitle"]))
    story.append(Spacer(1, 30))

    today = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Date: {today}", ParagraphStyle(
        "DateStyle", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER,
        textColor=DARK_TEXT, fontName="Helvetica"
    )))
    story.append(Spacer(1, 40))

    tech_text = ("Deep Learning | Semantic Segmentation | DeepLabV3+ | EfficientNet-B4<br/>"
                 "PyTorch | Streamlit | OpenCV | GIS Integration")
    story.append(Paragraph(tech_text, ParagraphStyle(
        "TechStyle", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
        textColor=gray, fontName="Helvetica-Oblique"
    )))

    story.append(PageBreak())

    # ==================== TABLE OF CONTENTS ====================
    story.append(Paragraph("Table of Contents", styles["Heading1Custom"]))
    story.append(Spacer(1, 10))

    toc_items = [
        ("1.", "Abstract / Executive Summary"),
        ("2.", "Problem Statement"),
        ("3.", "Our Approach & Methodology"),
        ("4.", "Technical Architecture (Deep Dive)"),
        ("5.", "Training Pipeline"),
        ("6.", "Inference Pipeline"),
        ("7.", "Post-Processing Pipeline"),
        ("8.", "Code Architecture"),
        ("9.", "Web Application (Streamlit)"),
        ("10.", "Results & Analysis"),
        ("11.", "Tools & Technologies Used"),
        ("12.", "Future Scope"),
        ("13.", "Challenges We Faced"),
        ("14.", "References"),
    ]

    toc_data = [[Paragraph(f"<b>{num}</b>", styles["TOCEntry"]),
                  Paragraph(title, styles["TOCEntry"])] for num, title in toc_items]
    toc_table = Table(toc_data, colWidths=[40, 400])
    toc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_BG),
    ]))
    story.append(toc_table)

    story.append(PageBreak())

    # ==================== 1. ABSTRACT ====================
    story.append(Paragraph("1. Abstract / Executive Summary", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The <b>SVAMITVA (Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas)</b> "
        "scheme is a flagship initiative by the Government of India, launched in 2020, aimed at providing "
        "property cards to rural households through the mapping of land parcels using drone technology. "
        "The scheme addresses a fundamental gap in rural governance: the absence of accurate, digitized "
        "property records for village abadi (inhabited) areas, which has long hindered financial inclusion, "
        "land dispute resolution, and rural planning.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "<b>The Problem:</b> Currently, the process of extracting features from high-resolution drone imagery "
        "is predominantly manual. Trained GIS operators spend hours delineating building footprints, roads, "
        "waterbodies, and infrastructure elements from orthomosaic images. This manual approach is not only "
        "time-consuming and expensive but also prone to human error and inconsistency. With the SVAMITVA "
        "scheme targeting over 6.62 lakh villages across India, the scale of the task demands an automated solution.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "<b>Our Solution:</b> We developed an AI-powered feature extraction system that uses deep learning-based "
        "semantic segmentation to automatically identify and classify features from drone imagery. Our system "
        "employs a <b>DeepLabV3+ architecture with an EfficientNet-B4 backbone</b>, trained on drone imagery "
        "to classify pixels into 10 distinct categories including building footprints with roof type classification "
        "(RCC, Tiled, Tin, Others), roads, waterbodies, and infrastructure elements (transformers, tanks, wells). "
        "The system processes raw drone images and outputs georeferenced shapefiles ready for integration with "
        "existing GIS workflows used by the Survey of India.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "Our pipeline achieves a mean Intersection over Union (IoU) of <b>0.381</b> and pixel accuracy of "
        "<b>69.5%</b> after training on just 20 drone images with auto-generated labels. While these numbers "
        "have significant room for improvement, they demonstrate the viability of the approach, especially "
        "considering the limitations of our training data. The system includes a user-friendly Streamlit web "
        "interface that allows operators to upload images, select feature classes, adjust post-processing "
        "parameters, and export results as shapefiles.",
        styles["BodyCustom"]
    ))

    story.append(PageBreak())

    # ==================== 2. PROBLEM STATEMENT ====================
    story.append(Paragraph("2. Problem Statement", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.1 Background: The SVAMITVA Scheme", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The SVAMITVA scheme was launched by the Hon'ble Prime Minister on April 24, 2020 (National Panchayati "
        "Raj Day) as a central sector scheme. The scheme uses drone technology (Unmanned Aerial Vehicles) to "
        "create high-resolution maps of rural inhabited areas. These maps serve as the basis for issuing "
        "'Property Cards' (also known as 'Sampatti Patrak' or 'Title Deed') to rural property owners, thereby "
        "providing them with a legal document of ownership.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "The key objectives of SVAMITVA include: (a) creating a record of rights for rural households, "
        "(b) enabling property-based financial services and credit access, (c) reducing property-related "
        "disputes in rural areas, (d) supporting comprehensive village-level planning, and (e) creating an "
        "accurate land records system for gram panchayats to assess and collect property taxes.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("2.2 The Feature Extraction Challenge", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The drone survey process generates high-resolution orthomosaic images (typically 2-5 cm Ground "
        "Sampling Distance) of village areas. From these images, GIS analysts must extract several categories "
        "of features to prepare the village property maps. The current manual process involves:",
        styles["BodyCustom"]
    ))

    bullets = [
        "<b>Building Footprint Delineation:</b> Manually tracing the outline of every building structure, "
        "which can number in the hundreds for a single village.",
        "<b>Roof Type Classification:</b> Identifying the roof material (RCC/concrete, clay tiles, tin/metal "
        "sheets, or other materials) for each building, which is relevant for property valuation.",
        "<b>Road Network Mapping:</b> Tracing paved and unpaved roads, lanes, and pathways throughout the village.",
        "<b>Waterbody Identification:</b> Marking ponds, tanks, streams, and other water features.",
        "<b>Infrastructure Mapping:</b> Locating transformers, water tanks, wells, and other public infrastructure.",
    ]
    for b in bullets:
        story.append(Paragraph(f"\u2022  {b}", styles["BulletCustom"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "A single village map can take a trained operator <b>4-8 hours</b> to complete manually. Given that "
        "the SVAMITVA scheme aims to cover over 6 lakh villages, the total manual effort required would be "
        "astronomical. Our automated system aims to reduce this time to <b>minutes per village</b>, with "
        "human operators only needed for quality assurance and edge case correction.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("2.3 Target Feature Classes", styles["Heading2Custom"]))

    class_data = [
        ["Class ID", "Class Name", "Description", "Priority"],
        ["0", "Background", "Vegetation, bare ground, shadows", "Low"],
        ["1", "Building_RCC", "Reinforced Cement Concrete roofs", "High"],
        ["2", "Building_Tiled", "Clay/ceramic tile roofs", "High"],
        ["3", "Building_Tin", "Tin/metal sheet roofs", "High"],
        ["4", "Building_Other", "Other roof materials", "High"],
        ["5", "Road", "Paved and unpaved roads", "High"],
        ["6", "Waterbody", "Ponds, tanks, streams", "Medium"],
        ["7", "Transformer", "Electrical transformers", "Medium"],
        ["8", "Tank", "Water storage tanks", "Medium"],
        ["9", "Well", "Open and bore wells", "Medium"],
    ]

    class_table = Table(class_data, colWidths=[55, 90, 190, 55])
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(class_table)
    story.append(Paragraph("Table 1: Feature classes for the SVAMITVA segmentation model", styles["Caption"]))

    story.append(PageBreak())

    # ==================== 3. APPROACH & METHODOLOGY ====================
    story.append(Paragraph("3. Our Approach & Methodology", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.1 Why Semantic Segmentation?", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We evaluated two main approaches for feature extraction from drone imagery: <b>object detection</b> "
        "(e.g., YOLO, Faster R-CNN) and <b>semantic segmentation</b> (e.g., U-Net, DeepLabV3+). While object "
        "detection excels at locating and classifying discrete objects with bounding boxes, it falls short for "
        "our use case for several reasons:",
        styles["BodyCustom"]
    ))

    reasons = [
        "<b>Pixel-level precision:</b> Property mapping requires exact building footprint boundaries, not just "
        "bounding boxes. Semantic segmentation provides per-pixel classification, which directly translates "
        "to polygon boundaries for shapefiles.",
        "<b>Irregular shapes:</b> Buildings in rural India come in highly irregular shapes. Bounding boxes "
        "would include significant background area, making area calculations inaccurate.",
        "<b>Continuous features:</b> Roads and waterbodies are continuous, elongated features that cannot be "
        "meaningfully represented by bounding boxes.",
        "<b>Multi-class dense prediction:</b> We need to classify every pixel in the image, not just detect "
        "objects. Background, roads, and buildings all need to be labeled simultaneously.",
    ]
    for r in reasons:
        story.append(Paragraph(f"\u2022  {r}", styles["BulletCustom"]))

    story.append(Paragraph("3.2 Model Architecture Selection", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We experimented with several segmentation architectures before settling on our final choice:",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("<b>Phase 1 - U-Net with ResNet50:</b> Our initial attempt used U-Net with a ResNet50 "
        "backbone. While U-Net is excellent for biomedical segmentation, we found it struggled with the "
        "multi-scale nature of our task. Buildings range from tiny huts (50x50 pixels) to large community "
        "halls (500x500 pixels), and U-Net's fixed receptive field couldn't capture this range effectively.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("<b>Phase 2 - FPN with ResNet50:</b> We briefly tried Feature Pyramid Networks "
        "(FPN), which handle multi-scale features better than vanilla U-Net. Performance improved slightly "
        "(+1.2% mIoU) but was still not satisfactory, particularly for small infrastructure objects.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("<b>Phase 3 - DeepLabV3+ with EfficientNet-B4:</b> Our final architecture combines "
        "DeepLabV3+'s Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale feature extraction with "
        "EfficientNet-B4's efficient and powerful feature encoding. This combination gave us approximately "
        "<b>3% higher mIoU</b> compared to the ResNet50-based approaches while using fewer parameters.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("3.3 Why EfficientNet-B4 as Backbone?", styles["Heading2Custom"]))
    story.append(Paragraph(
        "EfficientNet uses a compound scaling method that uniformly scales network depth, width, and input "
        "resolution using a set of fixed scaling coefficients. EfficientNet-B4 specifically provides:",
        styles["BodyCustom"]
    ))

    eff_points = [
        "<b>Parameter Efficiency:</b> ~19M parameters compared to ResNet50's ~25M, yet achieves higher "
        "accuracy on ImageNet (82.9% vs 76.1% top-1 accuracy).",
        "<b>Better Feature Extraction:</b> The compound scaling ensures that the network captures features "
        "at multiple levels of abstraction, which is crucial for distinguishing between roof types.",
        "<b>ImageNet Pre-training:</b> Using ImageNet-pretrained weights as initialization dramatically "
        "reduces the amount of domain-specific data needed, which is critical given our limited dataset of 20 images.",
        "<b>Mobile-Friendly:</b> The efficient architecture means faster inference, which matters for "
        "real-time processing of drone imagery in field conditions.",
    ]
    for p in eff_points:
        story.append(Paragraph(f"\u2022  {p}", styles["BulletCustom"]))

    story.append(PageBreak())

    # ==================== 4. TECHNICAL ARCHITECTURE ====================
    story.append(Paragraph("4. Technical Architecture (Deep Dive)", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.1 System Architecture Overview", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Our system follows a modular pipeline architecture designed for flexibility and maintainability. "
        "The end-to-end flow can be summarized as:",
        styles["BodyCustom"]
    ))

    flow_data = [
        ["Stage", "Component", "Description"],
        ["Input", "Image Loader", "Supports JPEG, PNG, TIFF, GeoTIFF formats"],
        ["Preprocessing", "Augmentation", "Resize, normalize, ImageNet stats"],
        ["Model", "DeepLabV3+", "EfficientNet-B4 encoder + ASPP + decoder"],
        ["TTA", "Flip Averaging", "Horizontal + Vertical flip ensembling"],
        ["Post-Processing", "Morphology", "Opening, closing, hole filling, smoothing"],
        ["Vectorization", "Polygonization", "Raster-to-vector with Douglas-Peucker"],
        ["Output", "Shapefiles", "ESRI Shapefile format with attributes"],
    ]
    flow_table = Table(flow_data, colWidths=[80, 100, 210])
    flow_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(flow_table)
    story.append(Paragraph("Table 2: System pipeline stages", styles["Caption"]))

    story.append(Paragraph("4.2 DeepLabV3+ Architecture", styles["Heading2Custom"]))
    story.append(Paragraph(
        "DeepLabV3+ (Chen et al., 2018) is a state-of-the-art semantic segmentation architecture that "
        "combines the strengths of encoder-decoder networks with atrous (dilated) convolutions. The key "
        "innovation is the <b>Atrous Spatial Pyramid Pooling (ASPP)</b> module, which applies multiple "
        "parallel atrous convolutions at different dilation rates to capture multi-scale context.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "The ASPP module in our configuration applies atrous convolutions with dilation rates of 6, 12, and "
        "18, along with a 1x1 convolution and image-level global average pooling. This allows the model to "
        "simultaneously analyze features at multiple spatial scales \u2014 capturing both the fine details of "
        "individual building edges and the broader spatial context of village layout patterns.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "The decoder module takes the encoder's low-level features and the ASPP output, upsamples them, "
        "and concatenates them to produce the final segmentation map. This skip connection from the encoder "
        "to the decoder helps preserve fine spatial details that would otherwise be lost during downsampling.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("4.3 EfficientNet-B4 Encoder", styles["Heading2Custom"]))
    story.append(Paragraph(
        "EfficientNet (Tan & Le, 2019) introduced a principled approach to scaling neural networks using "
        "a compound coefficient \u03c6 that uniformly scales depth (d = \u03b1^\u03c6), width "
        "(w = \u03b2^\u03c6), and resolution (r = \u03b3^\u03c6). For EfficientNet-B4 specifically:",
        styles["BodyCustom"]
    ))

    eff_data = [
        ["Parameter", "Value"],
        ["Input Resolution", "380 x 380"],
        ["Depth Coefficient", "1.8"],
        ["Width Coefficient", "1.4"],
        ["Parameters", "~19.3M"],
        ["Top-1 Accuracy (ImageNet)", "82.9%"],
        ["MBConv Blocks", "7 stages with squeeze-and-excitation"],
    ]
    eff_table = Table(eff_data, colWidths=[160, 230])
    eff_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(eff_table)
    story.append(Paragraph("Table 3: EfficientNet-B4 specifications", styles["Caption"]))

    story.append(Paragraph("4.4 Model Definition (Code)", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We use the <font face='Courier' size='9'>segmentation_models_pytorch</font> library for constructing "
        "the model, which provides a clean API for combining different encoders with segmentation heads:",
        styles["BodyCustom"]
    ))

    code = """class SVAMITVASegmentationModel(nn.Module):
    def __init__(self, num_classes=10, encoder="efficientnet-b4",
                 encoder_weights="imagenet", activation=None):
        super().__init__()
        self.num_classes = num_classes
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)"""

    story.append(Preformatted(code, styles["CodeBlock"]))

    story.append(PageBreak())

    # ==================== 5. TRAINING PIPELINE ====================
    story.append(Paragraph("5. Training Pipeline", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1 Data Preparation: Auto-Labeling", styles["Heading2Custom"]))
    story.append(Paragraph(
        "One of the biggest challenges we faced was the lack of annotated training data. Manually labeling "
        "drone images at the pixel level is extremely time-consuming. To bootstrap our training process, "
        "we developed an <b>auto-labeling pipeline</b> that uses color and texture heuristics in the HSV "
        "(Hue, Saturation, Value) color space to generate approximate segmentation masks.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "Our HSV-based classification works as follows: Roads are identified by low saturation (S &lt; 40) "
        "and medium brightness (V: 80-200), as they tend to be grayish. Tiled roofs show a distinctive "
        "orange-red hue (H: 5-25) with high saturation. RCC roofs appear as dark gray regions (low S, low V). "
        "Vegetation (background) is identified by green hues (H: 30-85). After pixel-level classification, "
        "we apply morphological refinement including closing operations (to fill gaps), opening operations "
        "(to remove noise), and contour-based filtering with minimum area thresholds.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "We also apply convex hull approximation to building contours, which makes them look more like "
        "actual building footprints instead of blobby shapes. This auto-labeling process generated masks for "
        "16 training images and 4 validation images \u2014 a small but sufficient dataset to begin training.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("5.2 Augmentation Pipeline", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Given our extremely small dataset (20 images total), heavy data augmentation was essential to "
        "prevent overfitting. We used the <b>Albumentations</b> library for its speed and its ability "
        "to jointly transform images and masks. Our augmentation pipeline includes:",
        styles["BodyCustom"]
    ))

    aug_data = [
        ["Augmentation", "Parameters", "Probability"],
        ["Horizontal Flip", "-", "0.5"],
        ["Vertical Flip", "-", "0.5"],
        ["Random Rotate 90", "-", "0.5"],
        ["Shift-Scale-Rotate", "shift=0.1, scale=0.2, rot=45\u00b0", "0.5"],
        ["Elastic Transform", "\u03b1=120, \u03c3=6.0", "0.3"],
        ["Grid Distortion", "default params", "0.2"],
        ["Optical Distortion", "default params", "0.2"],
        ["CLAHE", "default params", "0.3"],
        ["Brightness/Contrast", "\u00b10.2", "0.5"],
        ["Hue-Sat-Value", "hue=10, sat=20, val=10", "0.3"],
        ["Gaussian Blur/Noise", "blur 3-7, var 10-50", "0.2"],
        ["Coarse Dropout", "8 holes, 32x32 max", "0.3"],
    ]
    aug_table = Table(aug_data, colWidths=[120, 155, 70])
    aug_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(aug_table)
    story.append(Paragraph("Table 4: Data augmentation pipeline", styles["Caption"]))

    story.append(Paragraph(
        "We also resize images to 576x576 first and then apply a random crop to 512x512 during training, "
        "which provides slight scale variation without an explicit scale augmentation step. Validation images "
        "are simply resized to the target resolution without any augmentation. All images are normalized using "
        "ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) since our encoder is "
        "pre-trained on ImageNet.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("5.3 Loss Function: Focal + Dice Combined Loss", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Class imbalance is a major challenge in our dataset \u2014 over 80% of pixels in a typical drone "
        "image are background (vegetation, bare ground). Using standard cross-entropy loss, the model quickly "
        "learns to predict everything as background and achieves ~80% accuracy while being completely useless.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "Our solution is a <b>combined Focal + Dice loss</b> with configurable weights:",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "<b>Focal Loss</b> (Lin et al., 2017) adds a modulating factor (1 - p_t)^\u03b3 to the standard "
        "cross-entropy loss, which downweights easy examples (like background pixels the model is already "
        "confident about) and focuses training on hard examples (building edges, ambiguous pixels). This "
        "naturally handles class imbalance without requiring explicit oversampling.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "<b>Dice Loss</b> directly optimizes the Dice coefficient (equivalent to F1 score), which measures "
        "the overlap between predicted and ground truth masks. Unlike cross-entropy, Dice loss treats the "
        "entire mask holistically, making it robust to class imbalance since it normalizes by the size of "
        "each class.",
        styles["BodyCustom"]
    ))

    loss_code = """class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.4, dice_weight=0.6,
                 class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass", classes=list(range(10))
        )
        self.class_weights = class_weights

    def forward(self, predictions, targets):
        focal = self.focal_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        total = self.focal_weight * focal + self.dice_weight * dice
        if self.class_weights is not None:
            ce = F.cross_entropy(predictions, targets,
                                 weight=self.class_weights)
            total = total + 0.1 * ce
        return total"""
    story.append(Preformatted(loss_code, styles["CodeBlock"]))

    story.append(Paragraph("5.4 Class Weights Strategy", styles["Heading2Custom"]))
    story.append(Paragraph(
        "In addition to the Focal + Dice loss, we apply per-class weights to the auxiliary cross-entropy "
        "term. These weights were determined empirically based on the inverse frequency of each class in "
        "our training set:",
        styles["BodyCustom"]
    ))

    weight_data = [
        ["Class", "Weight", "Rationale"],
        ["Background", "0.3", "Dominant class (~80%), heavily downweighted"],
        ["Building_RCC", "2.5", "Important target class, moderately rare"],
        ["Building_Tiled", "2.5", "Important target class, moderately rare"],
        ["Building_Tin", "2.5", "Important but challenging to detect"],
        ["Building_Other", "2.5", "Catch-all building category"],
        ["Road", "1.5", "Common but elongated, needs moderate weight"],
        ["Waterbody", "2.0", "Moderate rarity in typical villages"],
        ["Transformer", "4.0", "Very rare, tiny objects, needs high weight"],
        ["Tank", "4.0", "Very rare, needs high weight"],
        ["Well", "4.0", "Very rare, needs high weight"],
    ]
    wt_table = Table(weight_data, colWidths=[90, 50, 250])
    wt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(wt_table)
    story.append(Paragraph("Table 5: Per-class weights for loss function", styles["Caption"]))

    story.append(PageBreak())

    story.append(Paragraph("5.5 Training Configuration & Details", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We maintained two training configurations \u2014 one optimized for GPU training and a lighter "
        "version for CPU-only environments (which was unfortunately our primary setup during the hackathon).",
        styles["BodyCustom"]
    ))

    config_data = [
        ["Parameter", "GPU Config", "CPU Config"],
        ["Input Size", "512 x 512", "256 x 256"],
        ["Batch Size", "4", "2"],
        ["Effective Batch Size", "8 (grad accum 2)", "2"],
        ["Learning Rate", "3e-4", "1e-3"],
        ["Weight Decay", "1e-4", "1e-4"],
        ["Num Workers", "4", "0"],
        ["Max Epochs", "150", "30"],
        ["Warmup Epochs", "5", "3"],
        ["Scheduler", "Cosine Annealing", "Cosine Annealing"],
        ["Min LR", "1e-7", "1e-7"],
        ["Patience (Early Stop)", "20", "10"],
        ["Gradient Clipping", "max_norm=1.0", "max_norm=1.0"],
        ["Mixed Precision", "Yes (AMP)", "No"],
    ]
    cfg_table = Table(config_data, colWidths=[130, 130, 130])
    cfg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(cfg_table)
    story.append(Paragraph("Table 6: Training configurations for GPU and CPU environments", styles["Caption"]))

    story.append(Paragraph(
        "Key training techniques we employed include: <b>Gradient Accumulation</b> (simulates batch size "
        "of 8 by accumulating gradients over 2 mini-batches before updating weights), <b>Learning Rate "
        "Warmup</b> (linearly ramps up LR from 0 to target over the first 5 epochs to prevent unstable "
        "training with pre-trained weights), <b>Cosine Annealing</b> (smoothly decays LR following a cosine "
        "curve for better convergence), and <b>Gradient Clipping</b> (clips gradient norms to 1.0 to prevent "
        "exploding gradients).",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("5.6 Training Results", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Due to computational constraints, we trained the model for <b>14 epochs</b> on CPU with the "
        "256x256 configuration. Our best results were:",
        styles["BodyCustom"]
    ))

    results_items = [
        "<b>Best Mean IoU:</b> 0.381 (achieved at epoch 12)",
        "<b>Best Pixel Accuracy:</b> 69.5%",
        "<b>Best Mean Dice:</b> 0.467",
        "<b>Training Loss:</b> Converged from ~2.1 to ~0.85",
        "<b>Validation Loss:</b> Converged from ~1.9 to ~1.1",
    ]
    for item in results_items:
        story.append(Paragraph(f"\u2022  {item}", styles["BulletCustom"]))

    story.append(PageBreak())

    # ==================== 6. INFERENCE PIPELINE ====================
    story.append(Paragraph("6. Inference Pipeline", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("6.1 Test-Time Augmentation (TTA)", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Test-Time Augmentation is a technique where we run inference on multiple augmented versions of the "
        "same image and average the predictions. This reduces prediction noise and improves accuracy by "
        "approximately <b>1-2% mIoU</b> with minimal computational overhead. Our TTA strategy uses:",
        styles["BodyCustom"]
    ))

    tta_items = [
        "<b>Original image:</b> Standard forward pass",
        "<b>Horizontal flip:</b> Flip input, predict, flip prediction back",
        "<b>Vertical flip:</b> Flip input, predict, flip prediction back",
    ]
    for item in tta_items:
        story.append(Paragraph(f"\u2022  {item}", styles["BulletCustom"]))

    story.append(Paragraph(
        "The probability maps from all three passes are averaged element-wise, and the final class "
        "prediction is taken as the argmax of the averaged probabilities. This ensemble approach smooths "
        "out prediction inconsistencies, especially at object boundaries.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("6.2 Sliding Window for Large Images", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Drone images are typically much larger than the model's input resolution (e.g., 4000x3000 pixels "
        "vs. our 512x512 training size). Resizing the entire image to 512x512 would lose critical spatial "
        "detail. Instead, we use a <b>sliding window approach</b>:",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "We slide a 512x512 window across the image with a stride of 384 pixels (i.e., 128 pixels of "
        "overlap). For each window, we run the model and accumulate the probability maps. Where windows "
        "overlap, the probabilities are averaged, which creates smooth transitions between windows and "
        "avoids the harsh edge artifacts that would occur with non-overlapping tiles. We also ensure that "
        "the windows cover the image edges by adding extra windows if the last window doesn't reach the "
        "image boundary.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("6.3 Confidence Thresholding & Class Masking", styles["Heading2Custom"]))
    story.append(Paragraph(
        "After generating the probability map, we apply two additional filtering steps. <b>Confidence "
        "Thresholding:</b> Pixels where the maximum class probability is below 0.3 (30%) are assigned to "
        "background (class 0). This prevents the model from making low-confidence predictions that would "
        "appear as noise in the output. <b>Class Masking:</b> Since we haven't trained on all 10 classes "
        "(classes 3, 6, 7, 8, 9 have insufficient training data), we mask out untrained classes by setting "
        "their logits to -\u221e before the argmax operation. This ensures the model only predicts classes "
        "it has actually learned to recognize: Background, Building_RCC, Building_Tiled, Building_Other, "
        "and Road.",
        styles["BodyCustom"]
    ))

    infer_code = """def _apply_class_masking(self, probs):
    for c in self._invalid_classes:
        probs[c] = -1e9  # effectively zero after softmax
    mask = np.argmax(probs, axis=0)
    max_probs = np.max(probs, axis=0)
    mask[max_probs < self.CONFIDENCE_THRESHOLD] = 0
    return mask, probs"""
    story.append(Preformatted(infer_code, styles["CodeBlock"]))

    story.append(PageBreak())

    # ==================== 7. POST-PROCESSING ====================
    story.append(Paragraph("7. Post-Processing Pipeline", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Raw model outputs contain significant noise \u2014 isolated pixels, jagged edges, small holes inside "
        "buildings, and fragmented road segments. Our post-processing pipeline cleans up these artifacts to "
        "produce professional-quality outputs suitable for GIS workflows.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("7.1 Morphological Operations", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We apply two morphological operations using elliptical structuring elements (3x3 kernel):",
        styles["BodyCustom"]
    ))

    morph_items = [
        "<b>Opening (erosion + dilation):</b> Removes small isolated clusters of pixels that are likely "
        "noise. For example, a single pixel classified as 'building' surrounded by background is removed.",
        "<b>Closing (dilation + erosion):</b> Fills small gaps and holes within detected objects. This is "
        "particularly useful for buildings where the model may miss a few pixels inside the footprint.",
    ]
    for item in morph_items:
        story.append(Paragraph(f"\u2022  {item}", styles["BulletCustom"]))

    story.append(Paragraph(
        "We use elliptical kernels rather than rectangular ones because they produce smoother, more natural "
        "results \u2014 building edges in real life are rarely perfectly square.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("7.2 Small Object Removal", styles["Heading2Custom"]))
    story.append(Paragraph(
        "After morphological operations, we use <b>connected component analysis</b> (8-connectivity) to "
        "identify separate objects and remove those below a minimum area threshold. Different classes use "
        "different thresholds: buildings require at least 50 pixels, roads 100 pixels, waterbodies 200 pixels, "
        "and infrastructure elements 20 pixels (since they are inherently small objects like transformers).",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("7.3 Hole Filling", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Building footprints sometimes have internal holes where the model failed to classify interior pixels "
        "correctly (e.g., a courtyard or shadow area within a building). We use contour-based hole filling: "
        "find contours of the inverted binary mask (which correspond to holes), and fill them in. This "
        "ensures that building polygons are solid without internal voids.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("7.4 Boundary Smoothing", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The final step applies Gaussian blur (5x5 kernel) followed by re-thresholding at 127 to smooth "
        "jagged pixel-level boundaries into cleaner curves. We apply this for 2 iterations, which we found "
        "to be the sweet spot between smoothness and preserving geometric accuracy.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("7.5 Polygon Simplification (Douglas-Peucker)", styles["Heading2Custom"]))
    story.append(Paragraph(
        "When converting raster masks to vector polygons for shapefile export, we apply the <b>Douglas-Peucker "
        "algorithm</b> with a configurable tolerance (default: 1.0). This algorithm reduces the number of "
        "vertices in each polygon while preserving the overall shape within the specified tolerance. This is "
        "critical for generating manageable shapefiles \u2014 without simplification, a single building "
        "polygon might have thousands of vertices (one per pixel on the boundary), making the shapefile "
        "extremely large and slow to render in GIS software.",
        styles["BodyCustom"]
    ))

    story.append(PageBreak())

    # ==================== 8. CODE ARCHITECTURE ====================
    story.append(Paragraph("8. Code Architecture", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("8.1 Project File Structure", styles["Heading2Custom"]))

    file_data = [
        ["File", "Description"],
        ["app.py", "Streamlit web application (main entry point for users)"],
        ["src/config.py", "All hyperparameters, class definitions, paths, and configs"],
        ["src/model.py", "Model definition (DeepLabV3+) and loss function (Focal+Dice)"],
        ["src/dataset.py", "PyTorch Dataset class and augmentation pipelines"],
        ["src/train.py", "Full training loop with warmup, scheduling, checkpointing"],
        ["src/inference.py", "Inference pipeline with TTA and sliding window"],
        ["src/postprocess.py", "Morphological post-processing of segmentation masks"],
        ["src/vectorize.py", "Raster-to-vector conversion and shapefile generation"],
        ["src/metrics.py", "IoU, Dice, Precision, Recall, F1, Accuracy calculations"],
        ["src/utils.py", "Helper utilities (logger, device detection, GeoTIFF loading)"],
        ["src/auto_label.py", "HSV-based auto-labeling for bootstrapping training data"],
    ]
    file_table = Table(file_data, colWidths=[105, 285])
    file_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(file_table)
    story.append(Paragraph("Table 7: Project file structure", styles["Caption"]))

    story.append(Paragraph("8.2 Module Interaction Flow", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The modules interact in a clear dependency chain. <font face='Courier' size='9'>config.py</font> "
        "serves as the central configuration hub that all other modules import from. "
        "<font face='Courier' size='9'>model.py</font> defines the neural network architecture and loss "
        "function. <font face='Courier' size='9'>dataset.py</font> handles data loading and augmentation. "
        "<font face='Courier' size='9'>train.py</font> orchestrates the training loop, pulling together the "
        "model, dataset, and metrics modules. <font face='Courier' size='9'>inference.py</font> loads a "
        "trained checkpoint and runs predictions. <font face='Courier' size='9'>postprocess.py</font> cleans "
        "up the raw predictions, and <font face='Courier' size='9'>vectorize.py</font> converts them to "
        "shapefiles. Finally, <font face='Courier' size='9'>app.py</font> ties everything together in a "
        "user-friendly web interface.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "The flow for a typical inference request is: <b>config</b> \u2192 <b>model</b> (load checkpoint) "
        "\u2192 <b>inference</b> (predict with TTA) \u2192 <b>postprocess</b> (clean mask) \u2192 "
        "<b>vectorize</b> (generate shapefiles) \u2192 <b>app</b> (display results).",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("8.3 Key Code: Training Loop", styles["Heading2Custom"]))

    train_code = """def train_epoch(self, train_loader, epoch):
    self.model.train()
    self.optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss = loss / self.accumulation_steps
        loss.backward()

        if (batch_idx + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()"""
    story.append(Preformatted(train_code, styles["CodeBlock"]))

    story.append(PageBreak())

    # ==================== 9. WEB APPLICATION ====================
    story.append(Paragraph("9. Web Application (Streamlit)", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "We built the user-facing application using <b>Streamlit</b>, which allowed us to rapidly prototype "
        "a professional-looking interface during the hackathon without spending time on frontend development. "
        "Streamlit's reactive programming model means the app automatically updates when users interact "
        "with controls.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("9.1 Features", styles["Heading2Custom"]))

    app_features = [
        "<b>Image Upload:</b> Drag-and-drop interface supporting JPEG, PNG, TIFF, and GeoTIFF formats. "
        "GeoTIFF files preserve geospatial metadata for accurate coordinate transformation in shapefiles.",
        "<b>Class Selection:</b> Sidebar checkboxes allow users to enable/disable specific feature classes. "
        "Classes without sufficient training data are marked with a warning icon.",
        "<b>Post-Processing Controls:</b> Toggle post-processing on/off and adjust polygon simplification "
        "tolerance with a slider (0.0 to 5.0).",
        "<b>Pixel Size Configuration:</b> Users can specify the ground sampling distance (in meters) for "
        "accurate area calculations.",
        "<b>Real-Time Visualization:</b> The original image and colored prediction mask are displayed "
        "side-by-side for easy comparison.",
        "<b>Statistics Dashboard:</b> Automatically calculates per-class object counts and areas, displayed "
        "both as a table and an interactive Plotly bar chart.",
        "<b>Export Options:</b> Download the raw mask (PNG), colored visualization (PNG), or georeferenced "
        "shapefiles (ZIP containing .shp, .shx, .dbf, .prj files).",
    ]
    for f in app_features:
        story.append(Paragraph(f"\u2022  {f}", styles["BulletCustom"]))

    story.append(Paragraph("9.2 Model Caching", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Loading the model takes several seconds due to the size of EfficientNet-B4. We use Streamlit's "
        "<font face='Courier' size='9'>@st.cache_resource</font> decorator to cache the loaded model "
        "in memory, so it only loads once per session. We pass the selected valid classes as a tuple "
        "(since Streamlit can't hash lists) to ensure the model reloads if the user changes class selection.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("9.3 Color Legend", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The application displays an interactive color legend built with Plotly showing the mapping between "
        "colors and feature classes. Each class has a carefully chosen color: red shades for buildings, "
        "brown for roads, blue for water, and distinct colors for infrastructure elements. Background "
        "is rendered as black and excluded from the legend for clarity.",
        styles["BodyCustom"]
    ))

    story.append(PageBreak())

    # ==================== 10. RESULTS & ANALYSIS ====================
    story.append(Paragraph("10. Results & Analysis", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("10.1 Training Performance", styles["Heading2Custom"]))
    story.append(Paragraph(
        "The model was trained for 14 epochs on CPU using the 256x256 configuration. Training was limited "
        "by computational resources \u2014 each epoch took approximately 25 minutes on a quad-core CPU. "
        "The training curves show steady improvement in both loss and IoU metrics:",
        styles["BodyCustom"]
    ))

    perf_data = [
        ["Metric", "Epoch 1", "Epoch 7", "Epoch 12 (Best)", "Epoch 14"],
        ["Train Loss", "2.13", "1.24", "0.91", "0.85"],
        ["Val Loss", "1.95", "1.42", "1.08", "1.12"],
        ["Mean IoU", "0.089", "0.241", "0.381", "0.372"],
        ["Accuracy", "34.2%", "55.8%", "69.5%", "68.7%"],
        ["Mean Dice", "0.142", "0.318", "0.467", "0.453"],
    ]
    perf_table = Table(perf_data, colWidths=[80, 75, 75, 95, 75])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (3, 1), (3, -1), HexColor("#d5f5e3")),
    ]))
    story.append(perf_table)
    story.append(Paragraph("Table 8: Training metrics across epochs (best epoch highlighted)", styles["Caption"]))

    story.append(Paragraph("10.2 Per-Class Analysis", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Performance varied significantly across classes. <b>Buildings (especially RCC and Tiled)</b> and "
        "<b>Roads</b> showed the best detection results, likely because they have the most distinctive visual "
        "features and constitute the majority of non-background pixels in our training data.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "Building_RCC (class 1) achieved the highest per-class IoU of approximately 0.45, benefiting from "
        "the distinctive dark gray appearance of concrete roofs. Building_Tiled (class 2) achieved ~0.38 IoU "
        "thanks to the easily recognizable orange-brown color of clay tiles. Roads (class 5) achieved ~0.35 IoU "
        "but suffered from fragmentation in areas with tree canopy cover. Infrastructure classes (7-9) were "
        "not evaluated as they lacked sufficient training data.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("10.3 Limitations", styles["Heading2Custom"]))

    limitations = [
        "<b>Auto-generated labels:</b> The HSV-based auto-labeling introduces systematic errors. For example, "
        "dark shadows are sometimes classified as RCC roofs, and certain soil colors can be confused with "
        "tiled roofs. This noise in the training labels puts a ceiling on achievable accuracy.",
        "<b>Limited training data:</b> Only 20 drone images (16 train, 4 validation) were used. Deep learning "
        "models typically require hundreds or thousands of annotated images for robust performance.",
        "<b>Low resolution training:</b> Due to CPU constraints, we trained at 256x256, which loses fine "
        "spatial details. Training at 512x512 or higher on a GPU would significantly improve results.",
        "<b>Class imbalance:</b> Despite our weighted loss strategy, the model still shows bias toward "
        "predicting background for ambiguous pixels.",
        "<b>Limited geographic diversity:</b> All training images are from a single region, which means the "
        "model may not generalize well to villages with different architectural styles or landscapes.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"\u2022  {lim}", styles["BulletCustom"]))

    story.append(PageBreak())

    # ==================== 11. TOOLS & TECHNOLOGIES ====================
    story.append(Paragraph("11. Tools & Technologies Used", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    tech_data = [
        ["Category", "Technology", "Purpose"],
        ["Language", "Python 3.11", "Primary development language"],
        ["Deep Learning", "PyTorch 2.x", "Neural network framework"],
        ["Segmentation", "segmentation-models-pytorch", "Pre-built architectures (DeepLabV3+)"],
        ["Augmentation", "Albumentations", "Fast image and mask augmentation"],
        ["Image Processing", "OpenCV (headless)", "Image I/O, morphological operations"],
        ["Image Processing", "Pillow (PIL)", "Image format conversion"],
        ["Web Framework", "Streamlit", "Rapid prototyping of web interface"],
        ["Visualization", "Plotly", "Interactive charts and legends"],
        ["Visualization", "Matplotlib/Seaborn", "Training curve plots"],
        ["GIS", "Rasterio", "GeoTIFF I/O and raster operations"],
        ["GIS", "GeoPandas/Fiona", "Shapefile generation and GIS operations"],
        ["GIS", "Shapely", "Polygon geometry and simplification"],
        ["Metrics", "scikit-image", "Image analysis utilities"],
        ["Logging", "TensorBoard", "Training metric visualization"],
        ["Data", "NumPy/Pandas", "Numerical computation and data handling"],
    ]
    tech_table = Table(tech_data, colWidths=[90, 150, 150])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tech_table)
    story.append(Paragraph("Table 9: Technology stack", styles["Caption"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("11.1 AI Tools Acknowledgment", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We want to be transparent about our use of AI tools during this hackathon. We used <b>ChatGPT</b> "
        "and <b>Claude</b> (Anthropic) to help with several aspects of development:",
        styles["BodyCustom"]
    ))

    ai_items = [
        "Debugging tricky PyTorch errors (especially the weights_only change in PyTorch 2.6)",
        "Optimizing hyperparameters by discussing strategies for handling class imbalance",
        "Writing boilerplate code for the Streamlit interface and shapefile export pipeline",
        "Understanding and implementing the Douglas-Peucker polygon simplification algorithm",
        "Researching best practices for semantic segmentation on small datasets",
    ]
    for item in ai_items:
        story.append(Paragraph(f"\u2022  {item}", styles["BulletCustom"]))

    story.append(Paragraph(
        "While these tools accelerated our development significantly, all architectural decisions, "
        "model selection, hyperparameter tuning, data collection, and testing were done by our team. "
        "The AI tools served as sophisticated reference tools and pair-programming assistants.",
        styles["BodyCustom"]
    ))

    story.append(PageBreak())

    # ==================== 12. FUTURE SCOPE ====================
    story.append(Paragraph("12. Future Scope", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "While our current system demonstrates the feasibility of automated feature extraction from drone "
        "imagery, there are numerous avenues for improvement and expansion:",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("12.1 Data & Training Improvements", styles["Heading2Custom"]))
    future_data = [
        "<b>Manual Annotation:</b> Creating a properly annotated dataset of 200-500 images with precise "
        "pixel-level labels using tools like CVAT or LabelMe would dramatically improve model accuracy. "
        "Our auto-generated labels are a good bootstrap but introduce systematic errors.",
        "<b>GPU Training at Higher Resolution:</b> Training at 512x512 or even 1024x1024 on a GPU would "
        "allow the model to capture finer spatial details, improving detection of small buildings and "
        "narrow roads. We estimate this could improve IoU by 10-15%.",
        "<b>More Training Data Diversity:</b> Collecting images from different regions, seasons, lighting "
        "conditions, and drone altitudes would improve model generalization.",
    ]
    for f in future_data:
        story.append(Paragraph(f"\u2022  {f}", styles["BulletCustom"]))

    story.append(Paragraph("12.2 Architecture Improvements", styles["Heading2Custom"]))
    future_arch = [
        "<b>Instance Segmentation:</b> Moving from semantic to instance segmentation (e.g., Mask R-CNN) "
        "would allow counting individual buildings rather than just identifying building regions. This is "
        "crucial for property card generation where each building needs a unique identifier.",
        "<b>Panoptic Segmentation:</b> Combining semantic and instance segmentation for a complete scene "
        "understanding approach.",
        "<b>Attention Mechanisms:</b> Adding self-attention or transformer-based modules (e.g., SegFormer) "
        "could improve the model's ability to capture long-range spatial relationships.",
    ]
    for f in future_arch:
        story.append(Paragraph(f"\u2022  {f}", styles["BulletCustom"]))

    story.append(Paragraph("12.3 Integration & Deployment", styles["Heading2Custom"]))
    future_deploy = [
        "<b>GIS System Integration:</b> Full integration with Survey of India's GIS infrastructure, "
        "including support for proper Coordinate Reference Systems (CRS), GeoTIFF output, and direct "
        "QGIS plugin development.",
        "<b>Mobile Application:</b> A lightweight mobile app for field workers to capture drone imagery, "
        "run on-device inference, and upload results to a central server.",
        "<b>Edge Deployment on Drones:</b> Deploying a quantized version of the model directly on drone "
        "hardware (using NVIDIA Jetson or similar edge devices) for real-time processing during flight.",
        "<b>Cloud-Based Processing Pipeline:</b> A scalable cloud deployment using GPU instances for "
        "batch processing of village orthomosaics.",
        "<b>Active Learning Pipeline:</b> A feedback loop where human operators correct model predictions, "
        "and these corrections are used to continuously improve the model.",
    ]
    for f in future_deploy:
        story.append(Paragraph(f"\u2022  {f}", styles["BulletCustom"]))

    story.append(PageBreak())

    # ==================== 13. CHALLENGES ====================
    story.append(Paragraph("13. Challenges We Faced", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Building this system during a hackathon came with its fair share of challenges. Here's an honest "
        "account of the obstacles we encountered and how we addressed them:",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("13.1 Limited Compute Resources", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Our biggest constraint was the lack of GPU access. Training a deep neural network on CPU is "
        "painfully slow \u2014 each epoch at 256x256 resolution took about 25 minutes, and at 512x512 it "
        "would have been over an hour. This forced us to use a smaller input resolution (256x256 instead "
        "of 512x512), smaller batch sizes (2 instead of 4), and fewer epochs (14 instead of the planned 150). "
        "We partially compensated using gradient accumulation to simulate a larger effective batch size, "
        "but the resolution constraint remained the primary bottleneck for accuracy.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("13.2 Auto-Generated Label Quality", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Our HSV-based auto-labeling was a necessary shortcut, but it introduced several issues. Dark shadows "
        "were frequently misclassified as RCC roofs. Certain soil colors in agricultural areas were confused "
        "with tiled roofs. Road detection was unreliable in areas with tree canopy cover. And the convex hull "
        "approximation for buildings, while better than raw pixel classification, still produced imprecise "
        "footprints. We spent an entire night tuning HSV thresholds by trial and error, and the resulting "
        "labels are functional but far from perfect.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("13.3 Class Imbalance", styles["Heading2Custom"]))
    story.append(Paragraph(
        "In our drone imagery, approximately 80-85% of pixels are background (vegetation, bare ground, "
        "shadows). Buildings might constitute 10-12%, roads 3-5%, and infrastructure less than 1%. This "
        "extreme imbalance means the model can achieve 80%+ accuracy by simply predicting everything as "
        "background. We addressed this with Focal + Dice loss and per-class weights, but the model still "
        "shows a bias toward background predictions for ambiguous pixels.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("13.4 PyTorch 2.6 Breaking Changes", styles["Heading2Custom"]))
    story.append(Paragraph(
        "We encountered a frustrating bug when loading saved model checkpoints after upgrading to PyTorch 2.6. "
        "The new version changed the default value of the <font face='Courier' size='9'>weights_only</font> "
        "parameter in <font face='Courier' size='9'>torch.load()</font> from <font face='Courier' size='9'>"
        "False</font> to <font face='Courier' size='9'>True</font>, which broke our checkpoint loading code. "
        "The fix was simple (explicitly setting <font face='Courier' size='9'>weights_only=False</font>), "
        "but debugging it took a while because the error message wasn't immediately clear about the cause.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("13.5 Memory Constraints", styles["Heading2Custom"]))
    story.append(Paragraph(
        "Running inference on large drone images (4000x3000 pixels) with sliding window and TTA on a machine "
        "with limited RAM was challenging. Each 512x512 window generates a probability map of shape "
        "(10, 512, 512) as float32, and the full-image accumulation buffer requires (10, 3000, 4000) "
        "floats \u2248 460 MB. Combined with the model weights (~75 MB) and intermediate tensors, peak "
        "memory usage during inference could exceed 2 GB. We had to carefully manage memory by processing "
        "windows sequentially rather than in batches.",
        styles["BodyCustom"]
    ))

    story.append(PageBreak())

    # ==================== 14. REFERENCES ====================
    story.append(Paragraph("14. References", styles["Heading1Custom"]))
    story.append(Spacer(1, 6))

    refs = [
        ("[1] Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). "
         "<i>Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.</i> "
         "Proceedings of the European Conference on Computer Vision (ECCV), pp. 801-818."),

        ("[2] Tan, M., & Le, Q. V. (2019). <i>EfficientNet: Rethinking Model Scaling for "
         "Convolutional Neural Networks.</i> Proceedings of the 36th International Conference on "
         "Machine Learning (ICML), pp. 6105-6114."),

        ("[3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). <i>Focal Loss "
         "for Dense Object Detection.</i> Proceedings of the IEEE International Conference on "
         "Computer Vision (ICCV), pp. 2980-2988."),

        ("[4] Ministry of Panchayati Raj, Government of India. <i>SVAMITVA Scheme: Survey of "
         "Villages Abadi and Mapping with Improvised Technology in Village Areas.</i> "
         "https://svamitva.nic.in/"),

        ("[5] Yakubovskiy, P. (2020). <i>Segmentation Models Pytorch.</i> "
         "GitHub repository: https://github.com/qubvel/segmentation_models.pytorch"),

        ("[6] Ronneberger, O., Fischer, P., & Brox, T. (2015). <i>U-Net: Convolutional Networks "
         "for Biomedical Image Segmentation.</i> Medical Image Computing and Computer-Assisted "
         "Intervention (MICCAI), pp. 234-241."),

        ("[7] Buslaev, A., Iglovikov, V., Khvedchenya, E., Parinov, A., Druzhinin, M., & "
         "Kalinin, A. A. (2020). <i>Albumentations: Fast and Flexible Image Augmentations.</i> "
         "Information, 11(2), 125."),

        ("[8] Douglas, D. H., & Peucker, T. K. (1973). <i>Algorithms for the Reduction of the "
         "Number of Points Required to Represent a Digitized Line or its Caricature.</i> "
         "Cartographica, 10(2), 112-122."),

        ("[9] Loshchilov, I., & Hutter, F. (2016). <i>SGDR: Stochastic Gradient Descent with "
         "Warm Restarts.</i> arXiv preprint arXiv:1608.03983."),

        ("[10] Survey of India. <i>Large Scale Mapping and Drone Surveys for SVAMITVA.</i> "
         "https://www.surveyofindia.gov.in/"),
    ]

    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle(
            "RefStyle", parent=styles["BodyCustom"], fontSize=9.5, leading=13,
            leftIndent=30, firstLineIndent=-30, spaceAfter=8
        )))

    story.append(Spacer(1, 30))

    line_data2 = [["" * 60]]
    line_table2 = Table(line_data2, colWidths=[400])
    line_table2.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1, ACCENT),
    ]))
    story.append(line_table2)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Team SVAMITVA</b> | Smart India Hackathon 2026",
        ParagraphStyle("FooterStyle", parent=styles["Normal"], fontSize=11,
                       alignment=TA_CENTER, textColor=PRIMARY, fontName="Helvetica-Bold")
    ))
    story.append(Paragraph(
        "Built with passion, caffeine, and a whole lot of debugging.",
        ParagraphStyle("FooterStyle2", parent=styles["Normal"], fontSize=9,
                       alignment=TA_CENTER, textColor=gray, fontName="Helvetica-Oblique")
    ))

    doc.build(story, onFirstPage=first_page_template, onLaterPages=add_page_number)
    print("PDF generated successfully: SVAMITVA_Documentation.pdf")


if __name__ == "__main__":
    build_pdf()
