import { createRouter, createWebHistory } from 'vue-router'
import KnowledgeBase from '../views/KnowledgeBase.vue'

const routes = [
  { path: '/', name: 'KnowledgeBase', component: KnowledgeBase },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
