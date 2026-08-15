import { expect, test } from '@playwright/test'

test('desktop workbench loads history and evidence inspector', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })
  await page.setViewportSize({ width: 1440, height: 960 })
  await page.goto('/')
  await expect(page.getByText('证据研究工作台')).toBeVisible()
  const runs = page.locator('.run-item')
  await expect(runs.first()).toBeVisible()
  await page.locator('.run-item', { hasText: '生成一份简短的装甲车辆毁伤评估标准' }).click()
  await expect(page.locator('.answer-document')).toBeVisible()
  await expect(page.locator('.answer-document table').first()).toBeVisible()
  await page.getByRole('button', { name: /原文/ }).click()
  const evidence = page.locator('.evidence-list > button')
  await expect(evidence.first()).toBeVisible()
  await evidence.first().click()
  await expect(page.locator('.evidence-detail blockquote')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  expect(overflow).toBeFalsy()
  expect(errors).toEqual([])
  await page.screenshot({ path: '../_inspect/web/desktop.png', fullPage: true })
})

test('mobile workbench uses panel navigation without overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: '历史' }).click()
  await expect(page.locator('.run-item').first()).toBeVisible()
  await page.locator('.run-item').first().click()
  await expect(page.locator('.answer-document')).toBeVisible()
  await page.getByRole('button', { name: /依据/ }).click()
  await expect(page.locator('.inspector-panel')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  expect(overflow).toBeFalsy()
  await page.screenshot({ path: '../_inspect/web/mobile.png', fullPage: true })
})

test('chapter research exposes evidence contribution and inference', async ({ page }) => {
  const run = {
    run_id: 'chapter-fixture', request: '生成毁伤评估标准', status: 'completed', route: 'supervisor',
    route_reason: '需要章节化研究', trace_id: null, evidence_count: 1, worker_count: 2,
    created_at: '2026-08-03T08:00:00Z', updated_at: '2026-08-03T08:01:00Z', error: null,
  }
  const evidence = {
    evidence_id: 'ev-levels', chunk_id: 'chunk-1', document_id: 'doc-1', source_file: '毁伤等级参考.pdf',
    page_start: 12, page_end: 12, section_path: ['毁伤分类'], quote: '装备功能状态可分为完好、受限、丧失和彻底毁坏。',
    quote_truncated: false, visual_assets: [], retrieval: [{ query: '毁伤等级分类', final_rank: 1, dense_rank: 1, dense_score: 0.9, bm25_rank: 2, bm25_score: 8, fusion_score: 0.03 }],
  }
  const planChapter = (chapter_id: string, ordinal: number, title: string, depends_on: string[] = []) => ({
    chapter_id, ordinal, title, objective: `${title}研究`, research_questions: ['如何形成规则'], depends_on,
    produces_contracts: chapter_id === 'principles' ? ['D-LEVELS'] : [], required_contracts: chapter_id === 'movement' ? ['D-LEVELS'] : [], acceptance_criteria: ['给出证据'],
  })
  const packet = (chapter_id: string, title: string, text: string) => ({
    task: `${title}研究`, chapter_id, chapter_title: title, depends_on: chapter_id === 'movement' ? ['principles'] : [], status: 'sufficient', summary: `${title}研究完成`,
    prose: text,
    rules: [{ basis: 'synthesized', evidence_ids: ['ev-levels'], rationale: '来源中的四种功能状态可形成四级可操作体系。', contract_id: chapter_id === 'principles' ? 'D-LEVELS' : null }],
    contracts: chapter_id === 'principles' ? [{ contract_id: 'D-LEVELS', type: 'terms', canonical_terms: ['完好', '受限', '丧失', '彻底毁坏'], applies_to_chapters: ['movement'] }] : [],
    conflicts: [], gaps: [], evidence_ids: ['ev-levels'],
  })
  const detail = {
    ...run, result: {
      run_id: run.run_id, request: run.request, route: { mode: 'supervisor', reason: run.route_reason },
      answer: { content: '# 标准', evidence_ids: ['ev-levels'], limitations: [] },
      document_plan: { title: '毁伤评估标准', rationale: '按依赖研究', chapters: [planChapter('principles', 1, '一、适用范围与总体原则'), planChapter('movement', 2, '二、运动能力评估标准', ['principles'])] },
      consistency_issues: [], evidence: [evidence], worker_packets: [packet('principles', '一、适用范围与总体原则', '本标准采用四级毁伤等级。'), packet('movement', '二、运动能力评估标准', '运动能力按统一四级体系判定。')], trace_id: null, created_at: run.created_at,
    },
  }
  await page.route('**/api/runs', async (route) => route.fulfill({ json: { items: [run] } }))
  await page.route('**/api/runs/chapter-fixture', async (route) => route.fulfill({ json: detail }))
  await page.goto('/')
  await page.locator('.run-item').click()
  await expect(page.getByRole('navigation', { name: '章节目录' })).toBeVisible()
  await expect(page.getByText('本标准采用四级毁伤等级。')).toBeVisible()
  await expect(page.getByRole('button', { name: /二、运动能力评估标准/ })).toBeVisible()
  await page.locator('.block-sources button').click()
  await expect(page.locator('.evidence-card.selected')).toBeVisible()
  await expect(page.locator('.evidence-card-quote')).toContainText('装备功能状态可分为完好、受限、丧失和彻底毁坏')
  await expect(page.getByText(/\[跨来源综合\]/)).toBeVisible()
  await page.screenshot({ path: '../_inspect/web/chapter-research.png', fullPage: true })
})
