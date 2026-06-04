// HoopsValue rankings — client-side sort + filter over the pre-rendered table.
// Rows ship in the HTML (instant first paint + SEO); this only adds interaction.
(function () {
  const table = document.getElementById('ranktable');
  const filter = document.getElementById('rankfilter');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const heads = Array.from(table.querySelectorAll('th'));
  let sortCol = 0, asc = true;

  if (filter) filter.addEventListener('input', () => {
    const q = filter.value.trim().toLowerCase();
    rows.forEach((r) => {
      const name = r.children[1].textContent.toLowerCase();
      r.style.display = !q || name.includes(q) ? '' : 'none';
    });
  });

  heads.forEach((th, i) => th.addEventListener('click', () => {
    const type = th.dataset.k;
    if (sortCol === i) asc = !asc;
    else { sortCol = i; asc = type === 'text'; }
    const sorted = rows.slice().sort((a, b) => {
      const ca = a.children[i], cb = b.children[i];
      let va, vb;
      if (type === 'num') { va = parseFloat(ca.dataset.v); vb = parseFloat(cb.dataset.v); }
      else { va = (ca.dataset.t || ca.textContent).toLowerCase(); vb = (cb.dataset.t || cb.textContent).toLowerCase(); }
      if (va < vb) return asc ? -1 : 1;
      if (va > vb) return asc ? 1 : -1;
      return 0;
    });
    sorted.forEach((r) => tbody.appendChild(r));
  }));
})();
