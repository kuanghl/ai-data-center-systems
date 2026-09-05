// translate.js (https://translate.zvo.cn) initialization for the VitePress site.
// Loaded right after the translate.js CDN script in the page <head>.
// NOTE: this file lives in static/ and is copied into public/ by scripts/prebuild.mjs.
(function () {
  var t = window.translate;
  if (!t) return;

  // Default free translation channel (api.translate.zvo.cn, no API key).
  // NOTE: client.edge was tried first but its auth endpoint
  // (edge.microsoft.com/translate/auth) 404s outside the Edge browser, so
  // every translation request fails. The default channel has a daily
  // character quota, which is fine for a docs site.
  t.service.use('translate.service');
  // Site content is English; keeps the service from auto-detecting the
  // visitor's browser language as the page language.
  t.language.setLocal('english');
  // Never translate typeset math: isIgnore() walks up the ancestor chain,
  // and every KaTeX output element carries the "katex" class.
  t.ignore.class.push('katex');
  // Only offer English / 简体中文 in the dropdown (v2 language ids, same
  // across all translation channels).
  t.selectLanguageTag.languages = 'english,chinese_simplified';
  // Render the dropdown into the nav-bar container instead of a div the
  // library would create at the end of <body>.
  t.selectLanguageTag.documentId = 'langSelect';
  // Watch dynamically changed DOM content (VitePress client-side navigation).
  t.listener.start();

  // translate.reset() (run on every language switch) disconnects the DOM
  // observer, and the library only re-arms it on the very first execute().
  // Re-arm it after each switch so SPA navigation keeps getting translated.
  t.lifecycle.changeLanguage.push(function () {
    setTimeout(function () {
      if (t.listener.use && !t.listener.isStart) {
        t.listener.addListener();
      }
    }, 2000);
  });

  // Render the built-in language dropdown into the nav-bar container
  // (#langSelect, see theme/Layout.vue). The library only renders it when
  // execute() runs, so trigger render() explicitly. Wait for the Vue app to
  // finish mounting first: VitePress hydrates asynchronously, and DOM
  // inserted into the nav bar before hydration gets wiped by the re-render.
  function mountSelect() {
    var mounted = function () {
      return !!document.querySelector('#app') && document.querySelector('#app').__vue_app__;
    };
    if (mounted()) {
      try {
        t.selectLanguageTag.render();
      } catch (e) {
        /* render() is idempotent; ignore */
      }
      return;
    }
    var timer = setInterval(function () {
      if (mounted()) {
        clearInterval(timer);
        try {
          t.selectLanguageTag.render();
        } catch (e) {
          /* render() is idempotent; ignore */
        }
      }
    }, 50);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountSelect);
  } else {
    mountSelect();
  }
})();
