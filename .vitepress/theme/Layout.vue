<script setup lang="ts">
import DefaultTheme from 'vitepress/theme';

const { Layout } = DefaultTheme;
</script>

<template>
  <Layout>
    <!-- translate-init.js renders the translate.js language select into this
         container (via selectLanguageTag.documentId) once the app is mounted. -->
    <template #nav-bar-content-after>
      <div id="langSelect" class="lang-select" />
    </template>
  </Layout>
</template>