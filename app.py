"""
POTENCIAL GRAVITACIONAL TERRESTRE
Geodesia Fisica | Universidad Distrital Francisco Jose de Caldas
Presentado para: Ing. Miguel Avila
"""
import math, os, json, threading
from flask import Flask, send_from_directory, request, jsonify, Response

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)

# ── Constantes GRS-80 / WGS-84 ────────────────────────────────────────────
KM    = 3.986004418e14          # constante grav. geocentrica [m3/s2]
a     = 6_378_137.0             # semieje mayor [m]
f     = 1.0 / 298.257222101     # achatamiento geometrico
omega = 7.292115e-5             # velocidad angular [rad/s]
W0    = 62_636_856.0            # potencial del geoide [J/kg]
gamma = 9.7976432222            # gravedad normal media [m/s2]

A2 = 1.62329e-3 * a**2
A3 = 2.29e-6    * a**3
A4 = 9.3e-6     * a**4
A5 = 2.3e-7     * a**5

# ── Calculo del potencial ─────────────────────────────────────────────────
def calcular(phi_deg: float) -> dict:
    phi  = math.radians(phi_deg)
    sp   = math.sin(phi); sp2 = sp**2; sp4 = sp**4
    s2p  = math.sin(2*phi); s2p2 = s2p**2
    r    = a * (1.0 - f * sp2)
    km_r = KM / r
    esf  = 1.0 + (A2/r**2) * (1/3 - sp2)
    ach  = (A3/r**3)*(2.5*sp2 - 1.5) + (A5/r**5)*(15/8 - 35/4*sp2 + 63/8*sp4)*sp
    asi  = (A4/r**4)*(3/35 + sp2/7 - 0.25*s2p2)
    ae   = km_r * esf
    aa   = km_r * ach
    aa2  = km_r * asi
    V    = ae + aa + aa2
    # Termino centrifugo y potencial total W = V + Z
    Z    = 0.5 * omega**2 * r**2 * math.cos(phi)**2
    W    = V + Z
    N    = (W - W0) / gamma      # undulacion zonal del geoide [m]
    return dict(
        phi=phi_deg, r=r, km_r=km_r,
        ae=ae, aa=aa, aa2=aa2,
        o0=km_r, o2=km_r*(A2/r**2)*(1/3-sp2), o35=aa, o4=aa2,
        V=V, W=W, Z=Z, N_zonal=N
    )

# ── Grilla de geoide para el heatmap ─────────────────────────────────────
def _tesseral_perturbation(lat_d: float, lon_d: float) -> float:
    """
    Perturbacion tesseral simplificada basada en los rasgos reales del geoide.
    Los centros/amplitudes aproximan las anomalias conocidas del EGM2008.
    Unidades: metros de undulacion.
    """
    lat = math.radians(lat_d)
    lon = math.radians(lon_d)

    def gauss(clat, clon, amp, sigma):
        dlat = math.radians(lat_d - clat)
        dlon = math.radians(lon_d - clon)
        # distancia angular aproximada
        d2 = dlat**2 + (math.cos(math.radians(clat)) * dlon)**2
        return amp * math.exp(-d2 / (2 * (math.radians(sigma))**2))

    # Rasgos principales del geoide real (lat, lon, amplitud[m], sigma[deg])
    features = [
        # Maximos
        ( -5,  147,  +86, 20),   # Nueva Guinea / Pacifico SO (maximo global)
        ( 65,  -20,  +64, 18),   # Islandia / Atlantico Norte
        ( 20,   20,  +58, 22),   # Norte de Africa / Mediteraneo
        ( 45,   90,  +45, 25),   # Asia Central
        (-30,  -30,  +40, 20),   # Atlantico Sur
        # Minimos
        (-10,   80, -107, 18),   # Oceano Indico Sur (minimo global)
        (-60,   20,  -70, 22),   # Antartico Indico
        (-45, -120,  -55, 20),   # Pacifico Sur
        ( 30, -100,  -40, 18),   # Mexico / Golfo
        ( 55,  160,  -35, 20),   # Kamchatka / Pacifico NW
    ]
    pert = sum(gauss(clat, clon, amp, sigma) for clat, clon, amp, sigma in features)

    # Harmonico C22 (el mayor tesseral real): amplitud ~10 m
    pert += 10.0 * math.cos(lat)**2 * math.cos(2*lon - math.radians(18))
    return pert

def build_geoid_grid() -> list:
    """Grilla 5° x 5° de undulacion del geoide N [m]."""
    grid = []
    V_ref = calcular(0)['V']
    for lat in range(-85, 86, 5):
        for lon in range(-180, 181, 5):
            d    = calcular(float(lat))
            Z    = 0.5 * omega**2 * d['r']**2 * math.cos(math.radians(lat))**2
            W    = d['V'] + Z
            N_z  = (W - W0) / gamma           # componente zonal [m]
            N_t  = _tesseral_perturbation(lat, lon)  # componente tesseral [m]
            N    = round(N_z + N_t, 3)
            grid.append({'lat': lat, 'lon': lon, 'N': N})
    return grid

_grid_cache = None
_grid_lock  = threading.Lock()

# ── NOAA proxy (un punto) ─────────────────────────────────────────────────
def query_noaa(lat: float, lon: float):
    try:
        import urllib.request
        url = (f'https://geodesy.noaa.gov/api/geoid/json'
               f'?lat={lat:.4f}&lon={lon:.4f}&model=14')
        req = urllib.request.Request(url,
              headers={'User-Agent': 'PotencialGravitacional/2.0'})
        r   = urllib.request.urlopen(req, timeout=8)
        d   = json.loads(r.read())
        h   = d.get('geoidHeight')
        if h is not None:
            return float(h), 'NOAA GEOID18'
    except Exception:
        pass
    return None, None

# ── LaTeX generator ───────────────────────────────────────────────────────
def generate_latex(phi_deg: float, hem: str, d: dict) -> str:
    s     = '-' if phi_deg < 0 else '+'
    phiA  = abs(phi_deg)
    hem_t = 'Sur' if hem == 'S' else 'Norte'

    def fmt(v, dec=4):
        return f"{v:,.{dec}f}".replace(',', 'X').replace('.', ',').replace('X', '.')

    lines = []
    W = lambda s: lines.append(s)

    W(r'\documentclass[12pt,a4paper]{article}')
    W(r'\usepackage[utf8]{inputenc}')
    W(r'\usepackage[T1]{fontenc}')
    W(r'\usepackage{amsmath,amssymb}')
    W(r'\usepackage{booktabs,array}')
    W(r'\usepackage[margin=2.5cm]{geometry}')
    W(r'\usepackage{xcolor,fancyhdr,titlesec}')
    W(r'\definecolor{nb}{RGB}{0,100,200}')
    W(r'\definecolor{nc}{RGB}{0,160,180}')
    W(r'\definecolor{no}{RGB}{210,80,20}')
    W(r'\pagestyle{fancy}\fancyhf{}')
    W(r'\fancyhead[L]{\small\color{nb}\textbf{Potencial Gravitacional Terrestre}}')
    W(r'\fancyhead[R]{\small\color{nb}GRS-80 / WGS-84}')
    W(r'\fancyfoot[C]{\small\thepage}')
    W(r'\renewcommand{\headrulewidth}{0.4pt}')
    W(r'\titleformat{\section}{\large\bfseries\color{nb}}{}{0em}{}[\titlerule]')
    W(r'\titleformat{\subsection}{\normalsize\bfseries\color{nc}}{}{0em}{}')
    W(r'\begin{document}')
    W(r'\begin{center}')
    W(r'{\LARGE\bfseries\color{nb}POTENCIAL GRAVITACIONAL TERRESTRE}\\[8pt]')
    W(r'{\large C\'alculo por Arm\'onicos Esf\'ericos Zonales}\\[4pt]')
    W(r'{\normalsize Modelo GRS-80\,/\,WGS-84\quad$\cdot$\quad Heiskanen \& Moritz (1967)}\\[12pt]')
    W(r'\hrule height 0.8pt\vspace{6pt}')
    W(r'{\normalsize\itshape Universidad Distrital Francisco Jos\'e de Caldas}\\')
    W(r'{\normalsize Ingenier\'ia Catastral y Geodesia\quad$\cdot$\quad Geodesia F\'isica}\\[4pt]')
    W(r'{\normalsize Presentado para: \textbf{Ing.\ Miguel Antonio \'Avila Angulo}}\\[4pt]')
    W(r'\hrule height 0.4pt')
    W(r'\end{center}\vspace{1em}')

    W(r'\section{Modelo Matem\'atico}')
    W(r'El potencial gravitacional exterior de la Tierra se expresa como')
    W(r'expansi\'on en arm\'onicos esf\'ericos zonales hasta el orden~5:')
    W(r'\begin{equation}\label{eq:V}')
    W(r'V(r,\varphi)=\frac{KM}{r}\Bigl[')
    W(r'  \underbrace{1+\frac{A_2}{r^2}\!\left(\tfrac{1}{3}-\sin^2\!\varphi\right)}_{\text{Esfera}+J_2}')
    W(r' +\underbrace{\frac{A_3}{r^3}\!\left(\tfrac{5}{2}\sin^2\!\varphi-\tfrac{3}{2}\right)')
    W(r'   +\frac{A_5}{r^5}\!\left(\tfrac{15}{8}-\tfrac{35}{4}\sin^2\!\varphi')
    W(r'   +\tfrac{63}{8}\sin^4\!\varphi\right)\sin\varphi}_{\text{Achatamiento}\;J_3,J_5}')
    W(r' +\underbrace{\frac{A_4}{r^4}\!\left(\tfrac{3}{35}+\tfrac{1}{7}\sin^2\!\varphi')
    W(r'   -\tfrac{1}{4}\sin^2\!2\varphi\right)}_{\text{Asimetr\'ia Ecuatorial}\;J_4}')
    W(r'\Bigr]\end{equation}')
    W(r'El potencial de la gravedad incluye el t\'ermino centr\'ifugo:')
    W(r'\begin{equation}')
    W(r'W(r,\varphi)=V(r,\varphi)+\tfrac{1}{2}\omega^2 r^2\cos^2\!\varphi')
    W(r'\end{equation}')
    W(r'La undulaci\'on del geoide (f\'ormula de Bruns) es:')
    W(r'\begin{equation}')
    W(r'N(\varphi)\approx\dfrac{W(\varphi)-W_0}{\bar\gamma}')
    W(r'\end{equation}')
    W(r'\subsection*{Radio parametrizado}')
    W(r'\begin{equation}r(\varphi)=a\!\left(1-f\sin^2\!\varphi\right)\end{equation}')

    W(r'\section{Constantes Geod\'esicas (GRS-80\,/\,WGS-84)}')
    W(r'\begin{center}\renewcommand{\arraystretch}{1.3}')
    W(r'\begin{tabular}{@{}lll@{}}')
    W(r'\toprule\textbf{S\'imbolo}&\textbf{Valor}&\textbf{Descripci\'on}\\\midrule')
    W(r'$KM$  & $3{,}986\,004\,418\times10^{14}$\,m$^3$\,s$^{-2}$ & Cte.\ grav.\ geocentrica\\')
    W(r'$a$   & $6\,378\,137{,}000$\,m & Semieje mayor del elipsoide\\')
    W(r'$f$   & $1/298{,}257\,222\,101$ & Achatamiento geom\'etrico\\')
    W(r'$\omega$ & $7{,}292\,115\times10^{-5}$\,rad\,s$^{-1}$ & Velocidad angular terrestre\\')
    W(r'$W_0$ & $62\,636\,856{,}0$\,J\,kg$^{-1}$ & Potencial del geoide\\')
    W(r'$A_2$ & $1{,}62329\times10^{-3}\,a^2$ & Coeficiente arm\'onico $J_2$\\')
    W(r'$A_3$ & $2{,}29\times10^{-6}\,a^3$   & Coeficiente arm\'onico $J_3$\\')
    W(r'$A_4$ & $9{,}3\times10^{-6}\,a^4$    & Coeficiente arm\'onico $J_4$\\')
    W(r'$A_5$ & $2{,}3\times10^{-7}\,a^5$    & Coeficiente arm\'onico $J_5$\\')
    W(r'\bottomrule\end{tabular}\end{center}')

    W(r'\section{C\'alculo Num\'erico para '
      f'$\\varphi = {phiA:.4f}^\\circ$ ({hem_t})'
      r'}')
    W(r'\subsection*{Paso 1 — Radio parametrizado}')
    W(r'\begin{align*}')
    W(f'r &= {a:.3f}\\;\\text{{m}}\\times\\left(1-\\tfrac{{1}}{{298{{,}}257}}\\,\\sin^2 {phiA:.4f}^\\circ\\right) = \\mathbf{{{fmt(d["r"],3)}\\;\\textbf{{m}}}}')
    W(r'\end{align*}')
    W(r'\subsection*{Paso 2 — T\'ermino de orden cero}')
    W(r'\begin{equation*}')
    W(f'\\frac{{KM}}{{r}} = \\frac{{3{{,}}986\\,004\\,418\\times10^{{14}}}}{{{fmt(d["r"],3)}}} = \\mathbf{{{fmt(d["km_r"],4)}\\;\\textbf{{J\\,kg}}^{{-1}}}}')
    W(r'\end{equation*}')
    W(r'\subsection*{Paso 3 — Desglose por \'ordenes arm\'onicos}')
    W(r'\begin{center}\renewcommand{\arraystretch}{1.4}')
    W(r'\begin{tabular}{@{}llr@{}}')
    W(r'\toprule\textbf{Orden}&\textbf{Componente}&\textbf{Aporte (J\,kg$^{-1}$)}\\\midrule')
    W(f'Orden~0  & Esfera pura $KM/r$              & ${fmt(d["o0"],4)}$\\\\')
    W(f'Orden~2  & Correc. $J_2$                   & ${d["o2"]:+.4f}$\\\\')
    W(f'Orden~3+5& Achatamiento $J_3,J_5$          & ${d["o35"]:+.6e}$\\\\')
    W(f'Orden~4  & Asimetr\'ia Ecuatorial $J_4$    & ${d["o4"]:+.4f}$\\\\')
    W(r'\midrule')
    W(f'\\textbf{{Total}} & $V=\\sum$ aportes & $\\mathbf{{{fmt(d["V"],4)}}}$\\\\')
    W(r'\bottomrule\end{tabular}\end{center}')
    W(r'\subsection*{Paso 4 — Potencial de la gravedad y undulaci\'on}')
    W(r'\begin{align*}')
    W(f'Z &= \\tfrac{{1}}{{2}}\\omega^2 r^2\\cos^2\\varphi = {fmt(d["Z"],4)}\\;\\text{{J\\,kg}}^{{-1}}\\\\')
    W(f'W &= V + Z = {fmt(d["W"],4)}\\;\\text{{J\\,kg}}^{{-1}}\\\\')
    W(f'N_{{\\text{{zonal}}}} &= (W - W_0)/\\bar\\gamma = {d["N_zonal"]:+.3f}\\;\\text{{m}}')
    W(r'\end{align*}')
    W(r'\subsection*{Resultado final}')
    W(r'\begin{equation*}\boxed{')
    W(f'V\\!\\left(r,{phiA:.4f}^\\circ\\right) = {d["V"]:.6e}\\;\\text{{J\\,kg}}^{{-1}}')
    W(r'}\end{equation*}')

    W(r'\section{Interpretaci\'on Geod\'esica}')
    W(r'\begin{itemize}')
    W(f'  \\item El \\textbf{{Orden~0}} (${fmt(d["o0"],2)}$\\,J\\,kg$^{{-1}}$) representa la masa puntual')
    W(r'        equivalente a toda la masa terrestre. Aporta $>99{,}9\,\%$ del total.')
    W(f'  \\item El \\textbf{{Orden~2}} (${d["o2"]:+.4f}$\\,J\\,kg$^{{-1}}$) corrige por el achatamiento')
    W(r'        polar ($J_2$ es el mayor de los arm\'onicos geof\'isicos).')
    W(f'  \\item El \\textbf{{Orden~3+5}} (${d["o35"]:+.4e}$\\,J\\,kg$^{{-1}}$) captura la')
    W(r'        asimetr\'ia norte-sur del geoide~(``pera geod\'esica\'\').')
    W(f'  \\item La \\textbf{{undulaci\'on zonal}} $N_{{\\text{{zonal}}}} = {d["N_zonal"]:+.3f}$\\,m representa la')
    W(r'        diferencia vertical entre el geoide y el elipsoide GRS-80 en esta latitud.')
    W(r'\end{itemize}')

    W(r'\begin{thebibliography}{9}')
    W(r'\bibitem{H67} W.\,A.~Heiskanen y H.~Moritz,')
    W(r'  \textit{Physical Geodesy}. W.\,H.~Freeman, 1967.')
    W(r'\bibitem{M80} H.~Moritz, ``GRS~1980\'\',')
    W(r'  \textit{Bull.\ G\'eod.}, 54:395--405, 1980.')
    W(r'\bibitem{T01} W.~Torge, \textit{Geodesy}, 3$^\text{rd}$~ed. de Gruyter, 2001.')
    W(r'\end{thebibliography}')
    W(r'\end{document}')

    return '\n'.join(lines)

# ── Rutas Flask ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "templates"), "index.html")

@app.route("/api/calcular", methods=["POST"])
def api_calcular():
    d   = request.get_json(force=True, silent=True) or {}
    phi = float(d.get("phi", 0))
    if d.get("hem","N").upper() == "S":
        phi = -abs(phi)
    return jsonify(calcular(phi))

@app.route("/api/tabla")
def api_tabla():
    return jsonify([calcular(float(p)) for p in range(91)])

@app.route("/api/geoide/grid")
def api_geoide_grid():
    global _grid_cache
    with _grid_lock:
        if _grid_cache is None:
            _grid_cache = build_geoid_grid()
    return jsonify(_grid_cache)

@app.route("/api/geoide/noaa")
def api_geoide_noaa():
    lat = float(request.args.get("lat", 0))
    lon = float(request.args.get("lon", 0))
    N, src = query_noaa(lat, lon)
    if N is None:
        d   = calcular(lat)
        N   = d["N_zonal"]
        src = "modelo-zonal"
    return jsonify({"lat": lat, "lon": lon, "N": round(N, 3), "source": src})

@app.route("/api/latex")
def api_latex():
    try:
        phi_deg = float(request.args.get("phi", 10))
        hem     = request.args.get("hem", "N").upper()
        if hem == "S":
            phi_deg = -abs(phi_deg)
        d = calcular(phi_deg)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    tex      = generate_latex(phi_deg, hem, d)
    phi_abs  = abs(phi_deg)
    filename = f"potencial_phi{phi_abs:.1f}{hem}.tex"
    return Response(tex, mimetype="text/plain",
                    headers={"Content-Disposition":
                             f'attachment; filename="{filename}"'})

if __name__ == "__main__":
    print("\n  POTENCIAL GRAVITACIONAL => http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
