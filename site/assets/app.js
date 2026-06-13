// HoopsValue static — instant client-side player search.
// Loads the pre-computed player list once (~tens of KB) and filters in-memory,
// so typing is instant with no server round-trip.
(function () {
  const input = document.getElementById('search');
  const box = document.getElementById('results');
  if (!input || !box) return;

  let players = [];
  let matches = [];
  let active = -1;

  const norm = (s) => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();

  fetch('data/players.json')
    .then((r) => r.json())
    .then((data) => { players = data; })
    .catch(() => {});

  function render() {
    if (!matches.length) { box.classList.remove('show'); box.innerHTML = ''; return; }
    box.innerHTML = matches.map((p, i) =>
      `<a href="player/${p.slug}.html" class="${i === active ? 'active' : ''}">
         <span class="nm">${p.name}</span>
         <span class="meta">${p.team} · ${p.pos} &nbsp; <span class="sc">${p.score}</span></span>
       </a>`).join('');
    box.classList.add('show');
  }

  function search(q) {
    const n = norm(q.trim());
    if (!n) { matches = []; active = -1; render(); return; }
    matches = players.filter((p) => norm(p.name).includes(n)).slice(0, 8);
    active = matches.length ? 0 : -1;
    render();
  }

  input.addEventListener('input', (e) => search(e.target.value));
  input.addEventListener('keydown', (e) => {
    if (!matches.length) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % matches.length; render(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + matches.length) % matches.length; render(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (matches[active]) location.href = `player/${matches[active].slug}.html`; }
    else if (e.key === 'Escape') { matches = []; render(); input.blur(); }
  });
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !box.contains(e.target)) { box.classList.remove('show'); }
  });
})();

// Legacy card picker — click a legend's name to show that career arc. The pick
// persists in sessionStorage so it survives navigation back to the home page.
(function () {
  var KEY = 'hv_legacy_pick';
  var wrap = document.querySelector('.legacy-picker-wrap');
  if (!wrap) return;
  var labels = [].slice.call(wrap.querySelectorAll('.lg-label'));
  var charts = [].slice.call(wrap.querySelectorAll('.lg-chart'));
  function show(idx) {
    labels.forEach(function (b) { b.classList.toggle('active', b.dataset.idx === idx); });
    charts.forEach(function (c) { c.classList.toggle('active', c.dataset.idx === idx); });
  }
  var saved = null;
  try { saved = sessionStorage.getItem(KEY); } catch (e) {}
  if (saved !== null && labels.some(function (b) { return b.dataset.idx === saved; })) show(saved);
  labels.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var idx = btn.dataset.idx;
      show(idx);
      try { sessionStorage.setItem(KEY, idx); } catch (e) {}
    });
  });
})();
