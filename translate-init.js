// translate.js (https://translate.zvo.cn) initialization for the VitePress site.
// Loaded right after the translate.js CDN script in the page <head>.
translate.selectLanguageTag.show = false; // hide the default floating language select; the site header provides its own switcher
translate.language.setLocal('english'); // site content is English
translate.service.use('client.edge'); // machine translation service channel
translate.listener.start(); // watch dynamically changed DOM content
// No translate.execute() here: the page stays in English until the user picks 简体中文 in the header switcher.
