// Exercise the live page with Node's built-in runner: node --test tests/live_ui.test.cjs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');

const html = fs.readFileSync(`${__dirname}/../mystery_dungeon/static/live.html`, 'utf8');

function page() {
  let focused;
  class Element {
    constructor(attrs = {}) {
      this.attrs = attrs;
      this.hidden = 'hidden' in attrs;
      this.textContent = '';
      this.children = [];
      this.handlers = {};
      this.style = {};
      this.dataset = {};
      this.value = '';
      this.position = 0;
      this.clientHeight = 200;
    }
    getAttribute(key) { return this.attrs[key]; }
    setAttribute(key, value) { this.attrs[key] = value; }
    addEventListener(key, handler) { this.handlers[key] = handler; }
    append(...nodes) {
      this.children.push(...nodes);
      for (const node of nodes) if (typeof node === 'object') node.parent = this;
    }
    remove() { this.parent.children = this.parent.children.filter(node => node !== this); }
    getBoundingClientRect() { return {height: nodes['live-view'].hidden ? 0 : 100}; }
    get scrollHeight() { return this.attrs.id === 'turn-feed' ? nodes['turn-messages'].children.length * 100 : 0; }
    get scrollTop() { return this.attrs.id === 'turn-feed' && nodes['live-view'].hidden ? 0 : this.position; }
    set scrollTop(value) { if (this.attrs.id === 'turn-feed' && nodes['live-view'].hidden) return; this.position = this.attrs.id === 'turn-feed' ? Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)) : value; }
    replaceChildren(...nodes) { this.children = nodes; }
    get childElementCount() { return this.children.length; }
    focus() { focused = this; }
    contains(node) { return this.children.includes(node); }
  }
  const nodes = {};
  for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const attrs = Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)]
      .map(m => [m[1], m[2]]));
    if (/\shidden[\s>]/.test(match[0])) attrs.hidden = '';
    nodes[match[1]] = new Element(attrs);
  }
  const tabs = Object.values(nodes).filter(node => node.attrs.role === 'tab');
  const context = vm.createContext({document: {
    get activeElement() { return focused; },
    getElementById: id => nodes[id],
    querySelectorAll: selector => selector === '[role="tab"]' ? tabs : [],
    createElement: () => new Element()
  }, setTimeout() {}, fetch: () => new Promise(() => {})});
  vm.runInContext(html.split('<script>')[1].split('</script>')[0], context);
  return {nodes, context, focused: () => focused,
    render(state) { context.sample = state; vm.runInContext('render(sample)', context); }};
}

function state() {
  return {status: 'thinking', running: true, game_state: {}, model: 'sample', output: 'sample',
    task: 'Explore', actions: ['A'], notes: ['Found: Exit'], findings: [], reset_count: 0,
    history: [{id: 1, event: 'action', screenshot: 'frames/000000.png', index: 0, action_number: 1, action: {type: 'press', button: 'A'}, source: 'agent',
      reasoning: '<b>Reasoning stays plain text</b>', task: 'Explore', model_ms: 10, tick_before: 20}]};
}

function key(node, key) { node.handlers.keydown({key, preventDefault() {}}); }

test('turns and notes support keyboard navigation and retain reading position', () => {
  const {nodes, focused, render} = page();
  render({...state(), history: actions(10)});
  nodes['turn-feed'].scrollTop = 250;
  nodes['turn-feed'].handlers.scroll();
  nodes['notes-tab'].handlers.click();
  assert.equal(nodes['live-view'].hidden, true);
  assert.equal(nodes['notes-view'].hidden, false);
  key(nodes['notes-tab'], 'ArrowRight');
  assert.equal(nodes['live-view'].hidden, false);
  assert.equal(nodes['turn-feed'].scrollTop, 250);
  assert.equal(focused(), nodes['live-tab']);
  key(nodes['live-tab'], 'End');
  assert.equal(focused(), nodes['notes-tab']);
  key(nodes['notes-tab'], 'Home');
  assert.equal(focused(), nodes['live-tab']);
  assert.equal(nodes['history-tab'], undefined);
});

function actions(count, start = 0) {
  return Array.from({length: count}, (_, i) => ({...state().history[0], index: start + i, id: start + i + 1,
    action_number: start + i + 1})).reverse();
}

test('turns append chronologically without replacing messages or open observations', () => {
  const {nodes, render} = page();
  render({...state(), history: actions(3)});
  const first = nodes['turn-messages'].children[0];
  assert.deepEqual(nodes['turn-messages'].children.map(node => node.dataset.id), ['1', '2', '3']);
  assert.equal(first.children[3].textContent, state().history[0].reasoning);
  first.children[4].open = true;
  render({...state(), history: actions(4)});
  assert.equal(nodes['turn-messages'].children.length, 4);
  assert.equal(nodes['turn-messages'].children[0], first);
  assert.equal(first.children[4].open, true);
  render({...state(), history: actions(4)});
  assert.equal(nodes['turn-messages'].children.length, 4);
});

test('the feed follows new turns only at the bottom and offers a jump back', () => {
  const {nodes, render, focused} = page();
  const feed = nodes['turn-feed'];
  render({...state(), history: actions(10)});
  assert.equal(feed.scrollTop, 800);
  render({...state(), history: actions(11)});
  assert.equal(feed.scrollTop, 900);
  feed.scrollTop = 200;
  feed.handlers.scroll();
  render({...state(), history: actions(12)});
  assert.equal(feed.scrollTop, 200);
  assert.equal(nodes['latest-button'].hidden, false);
  nodes['latest-button'].handlers.click();
  assert.equal(feed.scrollTop, 1000);
  assert.equal(nodes['latest-button'].hidden, true);
  assert.equal(focused(), feed);
});

test('trimming old messages preserves the reader position within surviving turns', () => {
  const {nodes, render} = page();
  const feed = nodes['turn-feed'];
  render({...state(), history: actions(50)});
  const kept = nodes['turn-messages'].children[3];
  feed.scrollTop = 350;
  feed.handlers.scroll();
  render({...state(), history: actions(50, 1)});
  assert.equal(nodes['turn-messages'].children.length, 50);
  assert.equal(nodes['turn-messages'].children[0].dataset.id, '2');
  assert.equal(nodes['turn-messages'].children.at(-1).dataset.id, '51');
  assert.equal(nodes['turn-messages'].children[2], kept);
  assert.equal(feed.scrollTop, 250);
});

test('new turns while viewing notes follow the latest when returning to the feed', () => {
  const {nodes, render} = page();
  render({...state(), history: actions(10)});
  nodes['notes-tab'].handlers.click();
  render({...state(), history: actions(15)});
  nodes['live-tab'].handlers.click();
  assert.equal(nodes['turn-feed'].scrollTop, 1300);
});

test('reset clears the feed and accepts fresh turns', () => {
  const {nodes, render} = page();
  render({...state(), history: actions(10)});
  render({...state(), reset_count: 1, status: 'paused', history: []});
  assert.equal(nodes['turn-messages'].children.length, 0);
  assert.equal(nodes['feed-empty'].hidden, false);
  assert.equal(nodes['turn-feed'].scrollTop, 0);
  assert.equal(nodes['latest-button'].hidden, true);
  render({...state(), reset_count: 1});
  assert.equal(nodes['turn-messages'].children.length, 1);
});

test('a disconnected server clears busy indicators and keeps polling', async () => {
  const {nodes, context, render} = page();
  render(state());
  let scheduled = false;
  context.fetch = async () => { throw new Error('offline'); };
  context.setTimeout = () => { scheduled = true; };
  await vm.runInContext('poll()', context);
  assert.equal(nodes['status-text'].textContent, 'Disconnected');
  assert.equal(nodes['thinking-line'].hidden, true);
  assert.equal(nodes['screen-activity'].hidden, true);
  assert.equal(scheduled, true);
});


test('trimming while Notes is open preserves the same surviving message on return', () => {
  const {nodes, render} = page();
  const feed = nodes['turn-feed'];
  render({...state(), history: actions(50)});
  const kept = nodes['turn-messages'].children[3];
  feed.scrollTop = 350;
  feed.handlers.scroll();
  nodes['notes-tab'].handlers.click();
  render({...state(), history: actions(50, 1)});
  assert.equal(nodes['turn-messages'].children[0].dataset.id, '1');
  nodes['live-tab'].handlers.click();
  assert.equal(nodes['turn-messages'].children[2], kept);
  assert.equal(feed.scrollTop, 250);
  assert.equal(nodes['latest-button'].hidden, false);
});

test('Stop turns append with their outcome, reasoning, and original observation', () => {
  const {nodes, render} = page();
  const stop = {event: 'stop', id: 2, outcome: 'stuck', reasoning: 'The exit is blocked.',
    source: 'agent', task: 'Explore', model_ms: 50, screenshot: 'frames/000007.png'};
  render(state());
  const first = nodes['turn-messages'].children[0];
  render({...state(), history: [stop, ...state().history]});
  const message = nodes['turn-messages'].children[1];
  assert.equal(nodes['turn-messages'].children[0], first);
  assert.equal(message.children[0].children[0].textContent, 'Turn #2');
  assert.equal(message.children[2].children[1].textContent, 'STOP · STUCK');
  assert.equal(message.children[3].textContent, 'The exit is blocked.');
  assert.equal(message.children[4].children[1].src, '/api/observation/000007');
});
