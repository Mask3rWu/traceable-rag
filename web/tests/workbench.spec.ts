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
  await page.getByRole('button', { name: /来源/ }).click()
  await expect(page.locator('.inspector-panel')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  expect(overflow).toBeFalsy()
  await page.screenshot({ path: '../_inspect/web/mobile.png', fullPage: true })
})
