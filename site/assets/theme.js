/* HoopsValue theme toggle. Light is the default (matching the app); the choice
   persists in localStorage. The FOUC guard in each page's <head> applies the
   stored theme before first paint; this file owns the toggle behaviour. */
(function () {
  var KEY = 'hv-theme';

  function current() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  window.hvToggleTheme = function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    if (next === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    try { localStorage.setItem(KEY, next); } catch (e) {}
  };
})();
