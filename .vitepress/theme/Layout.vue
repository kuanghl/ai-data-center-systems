<script setup lang="ts">
import { ref } from 'vue';
import DefaultTheme from 'vitepress/theme';

const { Layout } = DefaultTheme;
const translated = ref(false);

function toggleLanguage() {
  const t = (window as any).translate;
  if (!t) return;
  if (translated.value) {
    t.revert();
    translated.value = false;
  } else {
    t.execute('zh-CN');
    translated.value = true;
  }
}
</script>

<template>
  <Layout>
    <template #nav-bar-content>
      <button type="button" class="lang-switch" @click="toggleLanguage">
        {{ translated ? 'English' : '简体中文' }}
      </button>
    </template>
  </Layout>
</template>
