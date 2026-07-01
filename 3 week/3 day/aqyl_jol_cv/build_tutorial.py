# -*- coding: utf-8 -*-
"""Generate a detailed CV tutorial PDF (CNN, YOLOv8, detection vs classification)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, HRFlowable, Preformatted)

FIG = "docs/figs"
os.makedirs(FIG, exist_ok=True)
INK = colors.HexColor("#1b1e28"); AMBER = colors.HexColor("#d98407")
MUTED = colors.HexColor("#565d6e"); LINE = colors.HexColor("#e2e4ea")
CODEBG = colors.HexColor("#f3f4f7"); SOFT = colors.HexColor("#fff6e6")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.edgecolor": "#888"})

# --- register a Unicode (Cyrillic-capable) font: DejaVu, bundled with matplotlib ---
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
_fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", f"{_fd}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{_fd}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", f"{_fd}/DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuMono", f"{_fd}/DejaVuSansMono.ttf"))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

# ---------------------------------------------------------------- figures
def fig_conv():
    fig, ax = plt.subplots(figsize=(7.4, 3.1)); ax.axis("off")
    for (ox, title, n) in [(0, "Вход (пиксели)", 6), (5.2, "Feature map", 4)]:
        for i in range(n):
            for j in range(n):
                ax.add_patch(mp.Rectangle((ox + j * .32, 2 - i * .32), .3, .3,
                             fc="#eef1f7" if ox == 0 else "#fde9c8", ec="#b9c0cf"))
        ax.text(ox + n * .16, 2.55, title, ha="center", fontsize=10, color="#1b1e28")
    for i in range(3):  # highlighted 3x3 window
        for j in range(3):
            ax.add_patch(mp.Rectangle((j * .32, 2 - i * .32), .3, .3, fc="none", ec="#d98407", lw=2.2))
    ax.add_patch(mp.Rectangle((5.2, 2), .3, .3, fc="#f5a623", ec="#d98407", lw=2))
    ax.annotate("", xy=(5.15, 2.15), xytext=(1.0, 1.6),
                arrowprops=dict(arrowstyle="->", color="#d98407", lw=2))
    ax.text(3.0, .55, "Фильтр 3×3 скользит по входу и\nсчитает взвешенную сумму → одна клетка карты",
            ha="center", fontsize=9.5, color="#565d6e")
    ax.set_xlim(-.2, 7.2); ax.set_ylim(0, 3)
    plt.tight_layout(); plt.savefig(f"{FIG}/conv.png", bbox_inches="tight"); plt.close()

def fig_hierarchy():
    fig, ax = plt.subplots(figsize=(7.6, 1.9)); ax.axis("off")
    steps = ["Пиксели", "Края", "Текстуры", "Части", "Объект: машина"]
    for i, s in enumerate(steps):
        x = i * 1.5
        ax.add_patch(mp.FancyBboxPatch((x, 0), 1.25, .8, boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc="#fde9c8" if i == len(steps) - 1 else "#eef1f7", ec="#d98407" if i == len(steps)-1 else "#b9c0cf"))
        ax.text(x + .62, .4, s, ha="center", va="center", fontsize=9.5, color="#1b1e28")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.5, .4), xytext=(x + 1.25, .4),
                        arrowprops=dict(arrowstyle="->", color="#d98407", lw=1.8))
    ax.text(3.6, 1.15, "Чем глубже слой CNN — тем сложнее признаки", ha="center", fontsize=9.5, color="#565d6e")
    ax.set_xlim(-.2, 7.4); ax.set_ylim(-.2, 1.4)
    plt.tight_layout(); plt.savefig(f"{FIG}/hierarchy.png", bbox_inches="tight"); plt.close()

def fig_class_vs_detect():
    import glob
    snow = sorted(glob.glob("data/split/val/snow/*.jpg"))
    left = snow[0] if snow else "artifacts/road_gradcam.png"
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax[0].imshow(plt.imread(left)); ax[0].axis("off")
    ax[0].set_title("Классификация\nвся картинка → одна метка: «snow»", fontsize=10)
    ax[1].imshow(plt.imread("artifacts/road_damage_0.jpg")); ax[1].axis("off")
    ax[1].set_title("Детекция\nобъекты → боксы + тип + уверенность", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{FIG}/classvsdetect.png", bbox_inches="tight"); plt.close()

def fig_yolo_grid():
    img = plt.imread("site/assets/aqyljol_detect1.png")
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7.6, 7.6 * h / w)); ax.imshow(img); ax.axis("off")
    for gx in range(1, 8):
        ax.axvline(w * gx / 8, color="#ffb020", lw=.8, alpha=.55)
    for gy in range(1, 5):
        ax.axhline(h * gy / 5, color="#ffb020", lw=.8, alpha=.55)
    ax.set_title("YOLO делит кадр на сетку и в каждой ячейке предсказывает боксы + класс", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{FIG}/yolo_grid.png", bbox_inches="tight"); plt.close()

def fig_iou():
    fig, ax = plt.subplots(figsize=(4.6, 3.0)); ax.axis("off")
    ax.add_patch(mp.Rectangle((0.1, 0.3), 1.2, 1.0, fc="#cfe3ff", ec="#2f6fed", lw=2, alpha=.7))
    ax.add_patch(mp.Rectangle((0.7, 0.55), 1.2, 1.0, fc="#ffe0c2", ec="#d98407", lw=2, alpha=.7))
    ax.text(0.35, 1.42, "предсказание", color="#2f6fed", fontsize=9)
    ax.text(1.35, 1.62, "истина", color="#d98407", fontsize=9)
    ax.text(1.0, 0.05, "IoU = площадь пересечения / площадь объединения", ha="center", fontsize=9.5, color="#565d6e")
    ax.set_xlim(0, 2.1); ax.set_ylim(-.1, 1.8)
    plt.tight_layout(); plt.savefig(f"{FIG}/iou.png", bbox_inches="tight"); plt.close()

def fig_versions():
    fig, ax = plt.subplots(figsize=(7.6, 2.2)); ax.axis("off")
    vs = [("v1–v3", "16–18"), ("v4/v5", "20"), ("v6/v7", "22"),
          ("v8", "23"), ("v9/v10", "24"), ("v11", "24")]
    ax.plot([0, 6.2], [0.6, 0.6], color="#c7ccd8", lw=2)
    for i, (v, y) in enumerate(vs):
        x = 0.4 + i * 1.15
        hot = v == "v8"
        ax.plot(x, 0.6, "o", ms=16 if hot else 10, color="#d98407" if hot else "#aeb4c2")
        ax.text(x, 0.92 if hot else 0.85, "YOLO" + v, ha="center", fontsize=10 if hot else 9,
                color="#d98407" if hot else "#1b1e28", fontweight="bold" if hot else "normal")
        ax.text(x, 0.3, "20" + y, ha="center", fontsize=8, color="#565d6e")
    ax.text(2.0, 0.6 + .0, "", ha="center")
    ax.text(0.4 + 3 * 1.15, 1.25, "наш выбор", ha="center", fontsize=9, color="#d98407")
    ax.annotate("", xy=(0.4 + 3 * 1.15, 0.78), xytext=(0.4 + 3 * 1.15, 1.18),
                arrowprops=dict(arrowstyle="->", color="#d98407", lw=1.6))
    ax.set_xlim(-.2, 6.6); ax.set_ylim(0, 1.5)
    plt.tight_layout(); plt.savefig(f"{FIG}/versions.png", bbox_inches="tight"); plt.close()

for f in (fig_conv, fig_hierarchy, fig_class_vs_detect, fig_yolo_grid, fig_iou, fig_versions):
    f()
print("figures done")

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
def S(name, parent=None, **kw):
    return ParagraphStyle(name, parent=parent or ss["Normal"], **kw)
body = S("body", fontName="DejaVu", fontSize=10.3, leading=15.2, textColor=INK, spaceAfter=7)
bullet = S("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=4)
h1 = S("h1", fontName="DejaVu-Bold", fontSize=16.5, leading=20, textColor=INK, spaceBefore=6, spaceAfter=3)
h2 = S("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16, textColor=colors.HexColor("#a5670a"), spaceBefore=12, spaceAfter=4)
cap = S("cap", fontName="DejaVu-Oblique", fontSize=8.6, leading=11, textColor=MUTED, spaceAfter=12, alignment=1)
code = S("code", fontName="DejaVuMono", fontSize=8.6, leading=12, textColor=INK)
kick = S("kick", fontName="DejaVu-Bold", fontSize=9, textColor=AMBER, spaceAfter=2)
lead = S("lead", fontName="DejaVu", fontSize=11.5, leading=17, textColor=MUTED, spaceAfter=8)

story = []
def P(t, s=body): story.append(Paragraph(t, s))
def H1(t): story.extend([Spacer(1, 6), Paragraph(t, h1),
                         HRFlowable(width="100%", thickness=2, color=AMBER, spaceBefore=2, spaceAfter=8)])
def H2(t): P(t, h2)
def CAP(t): P(t, cap)
def CODE(t): story.extend([Preformatted(t, code), Spacer(1, 8)])
def FIGP(name, w=430):
    im = plt.imread(f"{FIG}/{name}")
    h, wd = im.shape[:2]
    story.append(Image(f"{FIG}/{name}", width=w, height=w * h / wd))
def BULLETS(items):
    for it in items:
        story.append(Paragraph("•&nbsp;&nbsp;" + it, bullet))
    story.append(Spacer(1, 5))

# ---------------------------------------------------------------- cover
story += [Spacer(1, 60)]
P("YDL 2026 · APPLIED DEEP LEARNING · TUTORIAL", kick)
story.append(Paragraph("Computer Vision в деталях", S("title", fontName="DejaVu-Bold", fontSize=30, leading=34, textColor=INK, spaceAfter=6)))
story.append(Paragraph("CNN, YOLOv8, детекция и классификация —<br/>как это устроено, на примере CV-слоя Aqyl&nbsp;Jol",
                       S("sub", fontName="DejaVu", fontSize=14, leading=20, textColor=MUTED, spaceAfter=20)))
story.append(HRFlowable(width="38%", thickness=3, color=AMBER, hAlign="LEFT", spaceAfter=16))
P("Три нейросети, которые «читают» дорогу: находят повреждения покрытия, определяют погоду и распознают "
  "транспорт. Этот документ объясняет, <b>что</b> под капотом и <b>почему</b> именно так — от пикселя до бокса.", lead)
story += [Spacer(1, 40)]
tbl = Table([["Что", "Архитектура", "Задача", "Результат"],
             ["Качество дороги", "YOLOv8", "детекция ям/трещин/люков", "mAP50 ≈ 0.40"],
             ["Состояние дороги", "ResNet18", "классификация погоды", "точность 0.82"],
             ["Транспорт", "YOLOv8", "детекция + тип машин", "~50 FPS (MPS)"]], colWidths=[110, 90, 160, 95])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONT", (0, 0), (-1, 0), "DejaVu-Bold", 9), ("FONT", (0, 1), (-1, -1), "DejaVu", 9),
    ("TEXTCOLOR", (0, 1), (-1, -1), INK), ("GRID", (0, 0), (-1, -1), .5, LINE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING", (0, 0), (-1, -1), 9)]))
story.append(tbl)
story.append(PageBreak())

# ---------------------------------------------------------------- 1. images as numbers
H1("1. Как компьютер «видит» картинку")
P("Для компьютера изображение — это <b>числа</b>. Цветная картинка размера H×W это тензор "
  "<b>H × W × 3</b>: три канала (красный, зелёный, синий), в каждой клетке — яркость пикселя от 0 до 255. "
  "Никакого «понимания» здесь нет — только сетка чисел. Задача нейросети: превратить эту сетку чисел в "
  "смысл («это машина», «здесь яма»).")
P("Почему нельзя просто взять обычную «полносвязную» сеть (как в Day 1)? Картинка 224×224×3 — это ~150 000 "
  "чисел. Если соединить каждый вход с каждым нейроном, параметров будут миллиарды, и сеть не увидит, что "
  "соседние пиксели связаны. Нужна архитектура, которая учитывает <b>структуру</b> картинки. Это CNN.")

# ---------------------------------------------------------------- 2. CNN
H1("2. CNN — свёрточная нейросеть")
P("<b>CNN</b> (Convolutional Neural Network) — сеть, специально придуманная для картинок. Все наши три "
  "модели — это CNN. В её основе три идеи:")
H2("2.1 Свёртка (convolution)")
P("Небольшой <b>фильтр</b> (ядро, например 3×3 числа) скользит по картинке и в каждом положении считает "
  "взвешенную сумму пикселей под собой. Результат — <b>карта признаков</b> (feature map), которая "
  "показывает, где в картинке встретился паттерн, «настроенный» в фильтре (например, вертикальный край). "
  "Ключевое: <b>фильтры не задаются вручную — сеть учит их сама</b> во время обучения.")
FIGP("conv.png", 400); CAP("Свёртка: фильтр 3×3 скользит по входу и формирует карту признаков.")
H2("2.2 Активация и pooling")
P("После свёртки применяется <b>ReLU</b> (обнуляет отрицательные значения — добавляет нелинейность), затем "
  "часто <b>pooling</b> — уменьшение карты (берём максимум в окне). Pooling делает сеть устойчивой к сдвигам "
  "и сокращает вычисления.")
H2("2.3 Иерархия признаков")
P("Слои CNN идут стопкой. Первые слои ловят простое (края, цвета), следующие — из краёв собирают текстуры, "
  "затем части объектов, а глубокие слои — целые объекты. Так из пикселей рождается смысл.")
FIGP("hierarchy.png", 430); CAP("От пикселей к объекту: чем глубже слой, тем сложнее признак.")
P("<b>Почему CNN хороша для картинок?</b> Два встроенных свойства (inductive bias):")
BULLETS(["<b>Локальность</b> — близкие пиксели обрабатываются вместе (объект — это локальный паттерн).",
         "<b>Инвариантность к сдвигу</b> — один и тот же фильтр ищет паттерн по всей картинке, поэтому "
         "машина распознаётся в любом углу кадра."])

# ---------------------------------------------------------------- 3. class vs detect
H1("3. Две задачи: классификация против детекции")
P("Обе наши «дорожные» темы — про CNN, но решают <b>разные</b> задачи. Важно понимать разницу.")
H2("Классификация")
P("Вход: вся картинка. Выход: <b>одна метка</b> на всё изображение + вероятности классов. "
  "Пример — наша погодная модель: фото дороги → «snow». Она не говорит <i>где</i> снег, она говорит "
  "<i>что</i> на всей сцене.")
H2("Детекция")
P("Вход: картинка. Выход: <b>список объектов</b>, для каждого — прямоугольник (bounding box) с координатами, "
  "класс и уверенность. Примеры — наши YOLO-модели: «вот здесь car 0.86», «вот здесь pothole 0.5». "
  "Детекция отвечает и <i>что</i>, и <i>где</i>.")
FIGP("classvsdetect.png", 430)
CAP("Слева — классификация (одна метка на кадр). Справа — детекция (боксы с типом и уверенностью).")

# ---------------------------------------------------------------- 4. classification model
H1("4. Классификация: ResNet18 + Transfer Learning")
P("Модель состояния дороги (погода) — это <b>ResNet18</b>, классическая CNN из 18 слоёв.")
H2("Что такое ResNet")
P("Глубокие сети трудно обучать: градиент «затухает». ResNet решает это <b>residual (skip) connections</b> — "
  "вход слоя добавляется к его выходу напрямую. Это позволяет строить очень глубокие, но обучаемые сети. "
  "ResNet18 — лёгкая версия, идеальна для ноутбука.")
H2("Transfer Learning — почему не учим с нуля")
P("Обучить CNN с нуля нужны миллионы картинок. Вместо этого берём ResNet18, <b>уже обученную на ImageNet</b> "
  "(1.2 млн изображений). Её «тело» (backbone) уже умеет извлекать общие признаки. Мы <b>замораживаем</b> "
  "backbone и заменяем только последний слой («голову») на свои 4 класса (fog/rain/snow/sand), после чего "
  "обучаем лишь голову. Достаточно ~1000 фото и нескольких минут.")
H2("Как происходит предсказание (пошагово)")
CODE("картинка → resize 224×224 → нормализация\n"
     "  → backbone ResNet18 → вектор признаков (512 чисел)\n"
     "  → Linear-слой → 4 логита\n"
     "  → softmax → вероятности [fog, rain, snow, sand]\n"
     "  → argmax → класс с макс. вероятностью")
P("Обучали 12 эпох, оптимизатор Adam, функция потерь cross-entropy. Итог: <b>точность 0.82</b> "
  "(snow 0.93, sand 0.87; fog и rain путаются — визуально похожи).")
H2("Grad-CAM — «куда смотрит сеть»")
P("Чтобы убедиться, что сеть решает по делу (а не по рамке кадра), используем <b>Grad-CAM</b> — тепловую "
  "карту важности пикселей. У нас она подсвечивает дорогу, небо и погодную текстуру — сеть «видит то, что надо».")
FIGP("../../artifacts/road_gradcam.png", 430)
CAP("Grad-CAM погодной модели: сверху — фото, снизу — внимание сети (последние две панели — уверенные ошибки).")

# ---------------------------------------------------------------- 5. detection YOLO
H1("5. Детекция: как работает YOLO")
P("<b>YOLO</b> = «You Only Look Once». Главная идея: <b>один прогон</b> сети сразу выдаёт все объекты кадра. "
  "Поэтому YOLO очень быстрая (real-time) — в отличие от старых двухэтапных детекторов, которые сначала "
  "искали регионы, потом их классифицировали.")
H2("Как это устроено")
BULLETS(["<b>Backbone</b> (CNN) извлекает признаки из картинки.",
         "<b>Neck</b> объединяет признаки разных масштабов (чтобы видеть и мелкие, и крупные объекты).",
         "<b>Head</b> на сетке ячеек предсказывает: координаты бокса, класс и «objectness» (есть ли объект)."])
FIGP("yolo_grid.png", 380)
CAP("YOLO мысленно делит кадр на сетку; каждая ячейка отвечает за объекты, чей центр в неё попал.")
H2("Порог уверенности и NMS")
P("Сеть выдаёт много боксов-кандидатов. Оставляем те, у кого уверенность выше порога (например 0.35), затем "
  "<b>NMS</b> (non-maximum suppression) убирает дубликаты — из нескольких перекрывающихся боксов на один "
  "объект оставляет лучший.")
H2("IoU — как измеряют «точность» бокса")
P("<b>IoU</b> (Intersection over Union) — насколько предсказанный бокс совпадает с настоящим: площадь "
  "пересечения делённая на площадь объединения. IoU = 1 — идеально; порог 0.5 — «считается попаданием».")
FIGP("iou.png", 260); CAP("IoU — мера пересечения предсказанного и истинного бокса.")

# ---------------------------------------------------------------- 6. why v8
H1("6. Почему именно YOLOv8 (а не другие версии)")
P("Версий YOLO много. Мы сознательно выбрали <b>YOLOv8</b> — вот контекст и обоснование.")
FIGP("versions.png", 430); CAP("Эволюция YOLO. v8 (Ultralytics, 2023) — зрелый, стабильный стандарт.")
tblv = Table([["Версия", "Год", "Авторы", "Особенность", "Anchor"],
              ["v1–v3", "2016–18", "Redmon", "родоначальник, Darknet", "anchor"],
              ["v4 / v5", "2020", "Bochkovskiy / Ultralytics", "PyTorch, массово популярный", "anchor"],
              ["v6 / v7", "2022", "Meituan / WongKinYiu", "выше скорость/точность", "anchor"],
              ["v8", "2023", "Ultralytics", "anchor-free, decoupled head — наш выбор", "anchor-free"],
              ["v9 / v10", "2024", "разные", "новее, эффективнее", "anchor-free"],
              ["v11", "2024", "Ultralytics", "последний, тот же API", "anchor-free"]],
             colWidths=[52, 58, 130, 165, 60])
tblv.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONT", (0, 0), (-1, 0), "DejaVu-Bold", 8.4), ("FONT", (0, 1), (-1, -1), "DejaVu", 8.2),
    ("BACKGROUND", (0, 4), (-1, 4), SOFT), ("FONT", (0, 4), (-1, 4), "DejaVu-Bold", 8.2),
    ("GRID", (0, 0), (-1, -1), .5, LINE), ("TEXTCOLOR", (0, 1), (-1, -1), INK),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
story += [tblv, Spacer(1, 10)]
H2("Почему v8 — правильный выбор для этого проекта")
BULLETS(["<b>Anchor-free + decoupled head.</b> v8 предсказывает боксы напрямую, без заранее заданных «якорей» "
         "(как в v5). Это проще, часто точнее и меньше ручной настройки.",
         "<b>Зрелый, стабильный API Ultralytics.</b> Всё обучение — три строки: <font face='DejaVuMono'>YOLO('yolov8n.pt').train(...)</font>. "
         "Минимум кода, максимум надёжности для учебного/демо проекта.",
         "<b>Баланс скорость/точность и размеры n/s/m/l/x.</b> Мы взяли самую лёгкую <b>nano</b> — она реально "
         "летает на ноутбуке (наши ~50 FPS на Apple MPS).",
         "<b>Работает на Apple Silicon (MPS).</b> Обучение и инференс — локально, без облака.",
         "<b>Огромное коммьюнити, готовые веса, документация.</b> Легко fine-tune (наш детектор дороги) и "
         "легко брать pretrained (наш детектор машин).",
         "<b>Совместимость с Aqyl&nbsp;Jol.</b> В статье используется YOLOv8/v11 — берём v8 как проверенный стандарт."])
H2("Почему не другие")
BULLETS(["<b>v3/v4 (Darknet)</b> — вне удобной PyTorch-экосистемы, сложнее дообучать.",
         "<b>v5</b> — отличный, но anchor-based и старее; при равном размере v8 обычно чуть точнее и проще.",
         "<b>v9/v10/v11</b> — новее и местами эффективнее, но v8 — «золотой стандарт»: максимально стабилен, "
         "предсказуем, отлично документирован. На нашем масштабе разница мала, а надёжность важнее. "
         "Перейти на v11 при желании — тот же API, одна строка."])

# ---------------------------------------------------------------- 7. metrics
H1("7. Как понять качество — метрики")
H2("Для классификации")
BULLETS(["<b>Accuracy</b> — доля верных ответов (у нас 0.82).",
         "<b>Precision / Recall / F1</b> — точность и полнота по каждому классу.",
         "<b>Confusion matrix</b> — где модель путается (у нас: fog ↔ rain)."])
H2("Для детекции")
BULLETS(["<b>IoU</b> — совпадение бокса с истиной.",
         "<b>mAP@50</b> — средняя точность при пороге IoU 0.5 (наш детектор дороги ≈ 0.40; Aqyl Jol 0.69–0.85).",
         "<b>mAP@50-95</b> — строже, среднее по порогам 0.5…0.95.",
         "<b>Precision-Recall кривая</b> — компромисс точность/полнота."])
tblm = Table([["Модель", "Задача", "Метрика", "Значение"],
              ["Качество дороги (YOLOv8)", "детекция", "mAP@50", "≈ 0.40"],
              ["Состояние дороги (ResNet18)", "классификация", "accuracy", "0.82"],
              ["Транспорт (YOLOv8)", "детекция", "скорость (MPS)", "~50 FPS"]], colWidths=[175, 95, 100, 75])
tblm.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONT", (0, 0), (-1, 0), "DejaVu-Bold", 9), ("FONT", (0, 1), (-1, -1), "DejaVu", 9),
    ("GRID", (0, 0), (-1, -1), .5, LINE), ("TEXTCOLOR", (0, 1), (-1, -1), INK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
story += [Spacer(1, 4), tblm, Spacer(1, 12)]

# ---------------------------------------------------------------- 8. pipeline & run
H1("8. Полный пайплайн и как запустить")
P("Каждая модель проходит один путь: <b>данные → обучение/готовая сеть → инференс → артефакт</b>.")
CODE("# окружение: conda env nn-lab (torch 2.12, MPS)\n"
     "python prepare_data.py       # DAWN → классы погоды\n"
     "python train_road.py          # ResNet18 (погода)\n"
     "python gradcam_road.py         # Grad-CAM\n"
     "python prepare_roaddamage.py   # Road Damage → YOLO-формат\n"
     "python train_road_damage.py    # YOLOv8 (повреждения)\n"
     "python infer_road_damage.py    # фото → боксы угроз\n"
     "python yolo_video.py           # YOLOv8 на видео (машины)")
P("Всё считается <b>локально на MacBook M1 (MPS)</b> — для летней школы этого достаточно, облако не нужно.")

# ---------------------------------------------------------------- summary
H1("Итог")
P("Все три модели — это <b>CNN</b>. Разница в задаче: <b>классификация</b> (одна метка на кадр — погода) и "
  "<b>детекция</b> (боксы с типом — повреждения дороги и транспорт). Для детекции мы взяли <b>YOLOv8</b> как "
  "быстрый, anchor-free, стабильный и хорошо документированный стандарт, совместимый с реальным проектом "
  "Aqyl&nbsp;Jol. Классификатор построен через <b>transfer learning</b> поверх ResNet18. Качество меряем "
  "accuracy (классификация) и mAP (детекция), а Grad-CAM даёт объяснимость.")
P("<font color='#a5670a'><b>Источники:</b></font> Redmon et al. «You Only Look Once» (2016); He et al. «Deep "
  "Residual Learning» (ResNet, 2015); Ultralytics YOLOv8 docs; Selvaraju et al. «Grad-CAM» (2017); "
  "Orynbassar et al. «Aqyl Jol» (2025).", S("src", parent=body, fontSize=9, textColor=MUTED))

# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8); canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "Computer Vision в деталях · Aqyl Jol Vision · YDL 2026")
    canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
    canvas.setStrokeColor(LINE); canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.restoreState()

doc = SimpleDocTemplate("CV_Tutorial.pdf", pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                        title="Computer Vision в деталях — Tutorial", author="Izbassar Orynbassar")
doc.build(story, onLaterPages=footer, onFirstPage=lambda c, d: None)
print("saved -> CV_Tutorial.pdf  (pages:", doc.page, ")")
