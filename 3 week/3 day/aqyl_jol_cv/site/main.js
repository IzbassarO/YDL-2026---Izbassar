/* ---------- animated perspective road (canvas) ---------- */
(function () {
  const cv = document.getElementById('road');
  const ctx = cv.getContext('2d');
  let W, H, dpr;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.width = innerWidth * dpr;
    H = cv.height = innerHeight * dpr;
  }
  resize();
  addEventListener('resize', resize);

  let offset = 0;
  const AMBER = 'rgba(255,176,32,';
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const vpx = W * 0.5, vpy = H * 0.30;

    // soft glow at the vanishing point
    const g = ctx.createRadialGradient(vpx, vpy, 0, vpx, vpy, H * 0.5);
    g.addColorStop(0, AMBER + '0.10)');
    g.addColorStop(1, 'rgba(255,176,32,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // converging lane lines
    ctx.lineWidth = 1 * dpr;
    for (let i = -7; i <= 7; i++) {
      const bx = vpx + i * (W * 0.16);
      ctx.strokeStyle = AMBER + (i === 0 ? '0.16)' : '0.05)');
      ctx.beginPath();
      ctx.moveTo(vpx, vpy);
      ctx.lineTo(bx, H);
      ctx.stroke();
    }

    // horizontal rows moving toward the viewer (driving forward)
    const N = 16;
    for (let k = 0; k < N; k++) {
      let p = ((k + offset) % N) / N;          // 0..1
      p = p * p;                                // perspective easing
      const yy = vpy + (H - vpy) * p;
      const half = (W * 0.92) * p;
      ctx.strokeStyle = AMBER + (0.03 + 0.10 * p).toFixed(3) + ')';
      ctx.lineWidth = (0.5 + 1.6 * p) * dpr;
      ctx.beginPath();
      ctx.moveTo(vpx - half, yy);
      ctx.lineTo(vpx + half, yy);
      ctx.stroke();
    }
    offset += 0.045;
    requestAnimationFrame(draw);
  }
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) draw();
})();

/* ---------- scroll progress + sticky nav ---------- */
const nav = document.getElementById('nav');
const progress = document.getElementById('progress');
addEventListener('scroll', () => {
  const st = scrollY, max = document.body.scrollHeight - innerHeight;
  progress.style.width = (max > 0 ? (st / max) * 100 : 0) + '%';
  nav.classList.toggle('scrolled', st > 40);
}, { passive: true });

/* ---------- reveal on scroll ---------- */
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
}, { threshold: 0.14 });
document.querySelectorAll('.reveal').forEach(el => io.observe(el));

/* ---------- active nav link ---------- */
const links = [...document.querySelectorAll('.nav-links a')];
const map = {};
links.forEach(a => { const id = a.getAttribute('href').slice(1); const s = document.getElementById(id); if (s) map[id] = a; });
const navIO = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      if (map[e.target.id]) map[e.target.id].classList.add('active');
    }
  });
}, { rootMargin: '-45% 0px -50% 0px' });
Object.keys(map).forEach(id => navIO.observe(document.getElementById(id)));

/* ---------- animated counters ---------- */
function animateCount(el) {
  const target = +el.dataset.count;
  const suffix = el.dataset.suffix || '';
  const dur = 1400, t0 = performance.now();
  function tick(now) {
    const p = Math.min((now - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = Math.round(target * eased);
    el.textContent = val.toLocaleString('en-US') + suffix;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
const countIO = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { animateCount(e.target); countIO.unobserve(e.target); } });
}, { threshold: 0.6 });
document.querySelectorAll('[data-count]').forEach(el => countIO.observe(el));
