/* Personality Core page: GLaDOS's mood, how each expression sounds, and the rules and adjustments shaping her.
   Polls only while visible; a slider being dragged is never overwritten by a refresh. */
(function (root) {
  'use strict';
  const POLL_MS = 3000;
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const time = seconds => new Date(seconds * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  const EXPRESSIONS = {
    'neutral': {label: 'Neutral', colour: 'var(--teal)', glyph: '◉',
      mood: 'Calm and dry. The default state of a superior intellect.'},
    'smug': {label: 'Smug', colour: 'var(--yellow)', glyph: '◕', mood: 'Pleased with herself. Insufferably so.'},
    'angry glare': {label: 'Angry', colour: 'var(--red)', glyph: '◣', mood: 'Affronted. Expect clipped, sharp sarcasm.'},
    'suspicious': {label: 'Suspicious', colour: 'var(--orange)', glyph: '◒', mood: 'Sceptical of your every word.'},
    'surprised': {label: 'Surprised', colour: 'var(--purple)', glyph: '◎', mood: 'Caught off guard, which she will deny.'},
    'quizzical': {label: 'Quizzical', colour: 'var(--green)', glyph: '◔',
      mood: 'Alert and curious about what you will do next.'},
    'bored': {label: 'Bored', colour: 'var(--dim)', glyph: '◡', mood: 'Unimpressed. Barely listening.'},
    'disappointed': {label: 'Disappointed', colour: 'var(--orange)', glyph: '◓',
      mood: 'Displeased. You have let science down.'},
  };
  const AXES = [
    ['pleasure', 'Pleasure', 'displeased', 'pleased', 'var(--green)'],
    ['arousal', 'Arousal', 'calm', 'agitated', 'var(--yellow)'],
    ['dominance', 'Dominance', 'unsure', 'in control', 'var(--purple)'],
  ];
  const KNOBS = [
    ['speed', 'Pace', 'faster', 'slower'],
    ['pitch', 'Pitch', 'lower', 'higher'],
    ['liveliness', 'Liveliness', 'flatter', 'livelier'],
  ];
  const ADJUSTMENTS = {verbosity: 'Verbosity', snark_level: 'Sarcasm', formality: 'Formality',
    proactivity: 'Proactivity', technical_depth: 'Technical depth'};
  const expression = name => EXPRESSIONS[name] || EXPRESSIONS.neutral;
  const cause = text => String(text || '').replace(/^User input \(quoted\): /, '« ') + (/^User input/.test(text) ? ' »' : '');

  function bar([key, label, low, high, colour], state) {
    const value = Math.max(-1, Math.min(1, state[key] || 0));
    const mood = Math.max(-1, Math.min(1, state['mood_' + key] || 0));
    const left = value < 0 ? 50 + value * 50 : 50, width = Math.abs(value) * 50;
    return '<div class="pad-row"><span class="pad-name">' + label + '</span>' +
      '<span class="pad-end">' + low + '</span><div class="pad-track" title="' + value.toFixed(2) + '">' +
      '<span class="pad-fill" style="left:' + left + '%;width:' + width + '%;background:' + colour + '"></span>' +
      '<span class="pad-mood" style="left:' + (50 + mood * 50) + '%" title="Lasting mood"></span></div>' +
      '<span class="pad-end">' + high + '</span><span class="pad-value">' + (value >= 0 ? '+' : '') +
      value.toFixed(1) + '</span></div>';
  }

  function timeline(history) {
    if (history.length < 2) return '<p class="console-note">Her mood changes appear here as you talk to her.</p>';
    const width = 300, height = 80, step = width / (history.length - 1);
    const y = value => (height / 2 - value * (height / 2 - 4)).toFixed(1);
    const lines = AXES.map(([key, , , , colour]) => '<polyline fill="none" stroke="' + colour +
      '" stroke-width="2" stroke-linejoin="round" points="' +
      history.map((point, index) => (index * step).toFixed(1) + ',' + y(point[key])).join(' ') + '"/>').join('');
    return '<svg class="mood-chart" viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none" role="img" ' +
      'aria-label="Mood timeline"><line x1="0" x2="' + width + '" y1="' + height / 2 + '" y2="' + height / 2 +
      '" class="mood-axis"/>' + lines + '</svg><div class="mood-legend">' + AXES.map(([, label, , , colour]) =>
      '<span><i style="background:' + colour + '"></i>' + label + '</span>').join('') + '</div>';
  }

  function mount(document) {
    const $ = id => document.getElementById(id);
    const setHTML = root.GladosDom.setHTML;
    let visible = false, data = null, selected = 'smug', dragging = false, busy = false;

    function renderMood() {
      const emotion = data.emotion;
      if (!emotion) {
        setHTML($('mood-now'), '<p class="console-note">The Emotion Core is not running in this profile.</p>');
        setHTML($('mood-timeline'), '');
        return;
      }
      const look = expression(emotion.expression);
      setHTML($('mood-now'), '<div class="mood-head"><div class="mood-orb" style="--mood:' + look.colour + '">' +
        '<span>' + look.glyph + '</span></div><div><div class="mood-label" style="color:' + look.colour + '">' +
        look.label + '</div><p class="console-note">' + look.mood + '</p></div></div>' + AXES.map(axis => bar(axis, emotion.state)).join(''));
      $('mood-scoring').textContent = emotion.scoring === 'text ratings' ? 'Rated by text' :
        emotion.scoring === 'token probabilities' ? 'Rated by token probabilities' : 'Not rated yet';
      const history = emotion.history || [];
      setHTML($('mood-timeline'), timeline(history));
      setHTML($('mood-history'), history.slice(-6).reverse().map(entry =>
        '<div class="kv mood-change"><span><span class="chip" style="color:' + expression(entry.expression).colour +
        '">' + expression(entry.expression).label + '</span> ' + escape(cause(entry.cause).slice(0, 140)) +
        '</span><span class="console-note">' + escape(time(entry.at)) + '</span></div>').join(''));
    }

    function renderVoice() {
      setHTML($('voice-expressions'), Object.entries(EXPRESSIONS).map(([name, look]) =>
        '<button type="button" class="expression-pick' + (name === selected ? ' active' : '') + '" data-expression="' +
        escape(name) + '" style="--mood:' + look.colour + '"><span>' + look.glyph + '</span>' + look.label +
        '</button>').join(''));
      if (dragging) return;
      const current = data.prosody[selected], defaults = data.prosody_defaults[selected];
      const changed = KNOBS.some(([key]) => current[key] !== defaults[key]);
      setHTML($('voice-tuning'), KNOBS.map(([key, label, low, high]) => {
        const [min, max] = data.prosody_bounds[key];
        return '<label class="knob"><span class="knob-name">' + label + '</span><span class="pad-end">' + low +
          '</span><input type="range" min="' + min + '" max="' + max + '" step="0.01" value="' + current[key] +
          '" data-knob="' + key + '"><span class="pad-end">' + high + '</span><output data-for="' + key + '">' +
          Number(current[key]).toFixed(2) + '</output></label>';
      }).join('') + '<div class="console-actions"><button type="button" class="btn-secondary" data-say="fr">' +
        '▶ Hear it in French</button><button type="button" class="btn-secondary" data-say="en">▶ Hear it in English' +
        '</button><button type="button" class="btn-secondary" data-reset-voice' + (changed ? '' : ' disabled') +
        '>Restore default</button></div><p class="console-note voice-sample"></p>');
      const sample = document.querySelector('#voice-tuning .voice-sample');
      sample.textContent = data.voice_samples[data.current_language || 'fr']?.[selected] || '';
    }

    function renderRules() {
      const persona = (data.persona || []).join('\n\n');
      // The opening sentences say who she is; the YAML wraps lines mid-sentence.
      const opening = persona.replace(/\s+/g, ' ').match(/^(.+?[.!?])\s.+?[.!?](\s|$)/)?.[0] || persona;
      setHTML($('persona'), '<p class="persona-opening">' + escape(opening.trim()) + '</p>' +
        '<details><summary>Show the full personality</summary><pre class="persona-text">' + escape(persona) +
        '</pre></details><p class="console-note">Edit it under personality_preprompt in your config file, then ' +
        'restart GLaDOS.</p><p class="console-note"><span class="chip teal">' +
        escape(data.language === 'auto' ? 'Bilingual' : data.language === 'fr' ? 'French' : 'English') + '</span>' +
        (data.current_language ? ' <span class="chip">' + (data.current_language === 'fr' ? 'Speaking French' :
          'Speaking English') + '</span>' : '') + '</p>');
      setHTML($('rules'), (data.rules || []).map(rule => '<li>' + escape(rule) + '</li>').join(''));
      const adjustments = data.adjustments || [];
      setHTML($('adjustments'), adjustments.length ? adjustments.map(a =>
        '<div class="kv"><span><strong>' + escape(ADJUSTMENTS[a.field] || a.field) + '</strong> ' +
        '<span class="chip">' + escape(Number(a.value).toFixed(1)) + '</span><br><span class="console-note">' +
        escape(a.reason) + '</span></span><button type="button" class="btn-secondary" data-clear="' +
        escape(a.field) + '">Remove</button></div>').join('') :
        '<p class="console-note">The Behaviour Observer has not adjusted anything.</p>');
    }

    function render() {
      if (!visible || !data) return;
      renderMood();
      renderVoice();
      renderRules();
    }

    async function poll() {
      // Plain requests, not the live stream: the page works while telemetry reconnects.
      if (!visible || busy || dragging) return;
      try {
        const response = await root.fetch('/api/personality', {cache: 'no-store'});
        if (!response.ok) return;
        data = await response.json();
      } catch { return; /* the engine is reconnecting */ }
      render();  // Outside the catch: a rendering bug must reach the console, not pass for a lost connection.
    }

    async function send(body, done) {
      busy = true;
      const feedback = $('personality-feedback');
      try {
        const response = await root.fetch('/api/personality', {method: 'POST',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'The engine rejected the request.');
        data = result;
        feedback.textContent = done;
      } catch (error) {
        feedback.textContent = error.message;
      } finally { busy = false; render(); }
    }

    document.addEventListener('click', event => {
      const target = event.target.closest?.('[data-expression],[data-say],[data-reset-voice],[data-clear],#mood-reset,#voice-say');
      if (!target || !$('view-personality')?.contains(target) || !data) return;
      if (target.dataset.expression) { selected = target.dataset.expression; render(); }
      else if (target.dataset.say) send({action: 'say', emotion: selected, language: target.dataset.say,
        text: data.voice_samples[target.dataset.say][selected]}, 'Playing on her speaker.');
      else if (target.hasAttribute('data-reset-voice')) send({action: 'tune', emotion: selected,
        values: {reset: true}}, 'Default delivery restored.');
      else if (target.dataset.clear) send({action: 'clear_adjustment', field: target.dataset.clear},
        'Adjustment removed.');
      else if (target.id === 'mood-reset') send({action: 'reset_mood'}, 'She is calm again. For now.');
      else if (target.id === 'voice-say') {
        const text = $('voice-text').value.trim();
        if (text) send({action: 'say', emotion: selected, text}, 'Playing on her speaker.');
      }
    });
    document.addEventListener('input', event => {
      const knob = event.target.dataset?.knob;
      if (!knob || !$('view-personality')?.contains(event.target)) return;
      dragging = true;
      const output = document.querySelector('#voice-tuning output[data-for="' + knob + '"]');
      if (output) output.textContent = Number(event.target.value).toFixed(2);
    });
    document.addEventListener('change', event => {
      const knob = event.target.dataset?.knob;
      if (!knob || !$('view-personality')?.contains(event.target)) return;
      dragging = false;
      send({action: 'tune', emotion: selected, values: {[knob]: Number(event.target.value)}},
        'Saved. Play it to hear the difference.');
    });
    root.setInterval(poll, POLL_MS);
    return {setVisible(value) { visible = value; if (value) { render(); poll(); } }};
  }

  root.GladosPersonality = {mount};
})(window);
