// === RANKING GRID ===

(function () {
  'use strict';
  var API_BASE = window.location.protocol + '//' + window.location.hostname + ':8889';

  // Season config
  const SEASONS = [
    { id: 1, label: 'T1', fullLabel: 'Temporada 1', year: '2022', episodes: 200 },
    { id: 2, label: 'T2', fullLabel: 'Temporada 2', year: '2023', episodes: 200 },
    { id: 3, label: 'T3', fullLabel: 'Temporada 3', year: '2024', episodes: 200 },
    { id: 4, label: 'T4', fullLabel: 'Temporada 4', year: '2025', episodes: 200 },
    { id: 5, label: 'T5', fullLabel: 'Temporada 5', year: '2026', episodes: 50 },
  ];

  // How many rows to show in the grid (max episodes across seasons for sample)
  const SAMPLE_EPISODES = 10;

  // DOM refs
  const gridHead = document.getElementById('grid-head');
  const gridBody = document.getElementById('grid-body');
  const gridFoot = document.getElementById('grid-foot');
  const modalOverlay = document.getElementById('modal-overlay');
  const modalTitle = document.getElementById('modal-title');
  const modalAvgValue = document.getElementById('modal-avg-value');
  const modalAvgVotes = document.getElementById('modal-avg-votes');
  const scoreButtonsContainer = document.getElementById('score-buttons');
  const modalComment = document.getElementById('modal-comment');
  const modalSubmit = document.getElementById('modal-submit');
  const modalStatus = document.getElementById('modal-status');
  const commentsList = document.getElementById('comments-list');
  const modalClose = document.getElementById('modal-close');

  // State
  let gridData = null; // { seasons: [ { id, episodes: [ { episode, avg, votes } ] } ] }
  let selectedScore = null;
  let currentCell = { season: null, episode: null };

  // ------ Color helpers ------

  function getScoreClass(score) {
    if (score == null || score === 0) return 'empty';
    if (score < 4) return 'score-bad';
    if (score < 6) return 'score-mediocre';
    if (score < 8) return 'score-ok';
    if (score < 9.5) return 'score-good';
    return 'score-great';
  }

  function getScoreColor(score) {
    if (score == null || score === 0) return '';
    if (score < 4) return '#d32f2f';
    if (score < 6) return '#f57c00';
    if (score < 8) return '#fbc02d';
    if (score < 9.5) return '#388e3c';
    return '#00e676';
  }

  // ------ Sample data generator ------

  function generateSampleData() {
    const data = { seasons: [] };
    SEASONS.forEach(function (s) {
      const eps = [];
      var count = Math.min(s.episodes, SAMPLE_EPISODES);
      for (var i = 1; i <= count; i++) {
        // Generate a somewhat realistic score distribution (mostly 5-9)
        var base = 5 + Math.random() * 4.5;
        // Occasional outliers
        if (Math.random() < 0.08) base = 2 + Math.random() * 2;
        if (Math.random() < 0.1) base = 9 + Math.random();
        var avg = Math.round(base * 10) / 10;
        avg = Math.min(10, Math.max(1, avg));
        eps.push({
          episode: i,
          avg: avg,
          votes: Math.floor(Math.random() * 50) + 1,
        });
      }
      data.seasons.push({ id: s.id, episodes: eps });
    });
    return data;
  }

  // ------ Fetch grid data ------

  async function fetchGridData() {
    try {
      var res = await fetch(API_BASE + '/api/ratings/grid');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var data = await res.json();
      return data;
    } catch (e) {
      console.warn('API not available, using sample data:', e.message);
      return null;
    }
  }

  // ------ Render grid ------

  function getMaxEpisodes(data) {
    var max = 0;
    data.seasons.forEach(function (s) {
      if (s.episodes.length > max) max = s.episodes.length;
    });
    return max;
  }

  function findEpisode(data, seasonId, epNum) {
    var season = data.seasons.find(function (s) { return s.id === seasonId; });
    if (!season) return null;
    return season.episodes.find(function (e) { return e.episode === epNum; }) || null;
  }

  function renderGrid(data) {
    gridData = data;
    var maxEp = getMaxEpisodes(data);

    // Header
    var headHtml = '<tr><th></th>';
    SEASONS.forEach(function (s) {
      headHtml += '<th title="' + s.fullLabel + ' (' + s.year + ')">' + s.label + '</th>';
    });
    headHtml += '</tr>';
    gridHead.innerHTML = headHtml;

    // Body
    var bodyHtml = '';
    for (var ep = 1; ep <= maxEp; ep++) {
      bodyHtml += '<tr><td>E' + ep + '</td>';
      SEASONS.forEach(function (s) {
        var epData = findEpisode(data, s.id, ep);
        if (epData && epData.avg > 0) {
          var cls = getScoreClass(epData.avg);
          bodyHtml += '<td><div class="rating-cell ' + cls + '" '
            + 'data-season="' + s.id + '" data-episode="' + ep + '">'
            + epData.avg.toFixed(1)
            + '<span class="cell-tooltip">' + s.fullLabel + ' - Ep ' + ep
            + ' | ' + epData.avg.toFixed(1) + '/10 (' + epData.votes + ' votos)</span>'
            + '</div></td>';
        } else if (ep <= (s.episodes || SAMPLE_EPISODES)) {
          bodyHtml += '<td><div class="rating-cell empty" '
            + 'data-season="' + s.id + '" data-episode="' + ep + '">'
            + '-'
            + '<span class="cell-tooltip">' + s.fullLabel + ' - Ep ' + ep + ' | Sin votos</span>'
            + '</div></td>';
        } else {
          bodyHtml += '<td></td>';
        }
      });
      bodyHtml += '</tr>';
    }
    gridBody.innerHTML = bodyHtml;

    // Footer (averages)
    var footHtml = '<tr><td>Prom.</td>';
    SEASONS.forEach(function (s) {
      var seasonData = data.seasons.find(function (sd) { return sd.id === s.id; });
      if (seasonData && seasonData.episodes.length > 0) {
        var scored = seasonData.episodes.filter(function (e) { return e.avg > 0; });
        if (scored.length > 0) {
          var sum = scored.reduce(function (acc, e) { return acc + e.avg; }, 0);
          var avg = sum / scored.length;
          var cls = getScoreClass(avg);
          footHtml += '<td class="' + cls + '" style="color: ' + getScoreColor(avg) + '">'
            + avg.toFixed(1) + '</td>';
        } else {
          footHtml += '<td>-</td>';
        }
      } else {
        footHtml += '<td>-</td>';
      }
    });
    footHtml += '</tr>';
    gridFoot.innerHTML = footHtml;

    // Attach click handlers
    var cells = gridBody.querySelectorAll('.rating-cell');
    cells.forEach(function (cell) {
      cell.addEventListener('click', function () {
        var season = parseInt(cell.getAttribute('data-season'));
        var episode = parseInt(cell.getAttribute('data-episode'));
        openModal(season, episode);
      });
    });
  }

  // ------ Modal ------

  function openModal(season, episode) {
    currentCell = { season: season, episode: episode };
    selectedScore = null;

    var seasonCfg = SEASONS.find(function (s) { return s.id === season; });
    modalTitle.textContent = (seasonCfg ? seasonCfg.fullLabel : 'Temporada ' + season) + ' - Programa ' + episode;

    // Show current average
    var epData = findEpisode(gridData, season, episode);
    if (epData && epData.avg > 0) {
      modalAvgValue.textContent = epData.avg.toFixed(1);
      modalAvgVotes.textContent = '(' + epData.votes + ' votos)';
    } else {
      modalAvgValue.textContent = '--';
      modalAvgVotes.textContent = '(0 votos)';
    }

    // Build score buttons
    scoreButtonsContainer.innerHTML = '';
    for (var i = 1; i <= 10; i++) {
      var btn = document.createElement('button');
      btn.className = 'score-btn';
      btn.textContent = i;
      btn.setAttribute('data-score', i);
      btn.addEventListener('click', function () {
        selectedScore = parseInt(this.getAttribute('data-score'));
        var allBtns = scoreButtonsContainer.querySelectorAll('.score-btn');
        allBtns.forEach(function (b) { b.classList.remove('selected'); });
        this.classList.add('selected');
      });
      scoreButtonsContainer.appendChild(btn);
    }

    // Clear comment & status
    modalComment.value = '';
    modalStatus.textContent = '';
    modalStatus.className = 'modal-status';

    // Load comments
    loadComments(season, episode);

    modalOverlay.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    modalOverlay.classList.remove('open');
    document.body.style.overflow = '';
  }

  modalClose.addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', function (e) {
    if (e.target === modalOverlay) closeModal();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && modalOverlay.classList.contains('open')) closeModal();
  });

  // ------ Submit rating ------

  modalSubmit.addEventListener('click', async function () {
    if (selectedScore == null) {
      modalStatus.textContent = 'Selecciona un puntaje antes de enviar.';
      modalStatus.className = 'modal-status error';
      return;
    }

    modalSubmit.disabled = true;
    modalStatus.textContent = 'Enviando...';
    modalStatus.className = 'modal-status';

    var payload = {
      season: currentCell.season,
      episode: currentCell.episode,
      score: selectedScore,
      comment: modalComment.value.trim(),
    };

    try {
      var res = await fetch(API_BASE + '/api/ratings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('HTTP ' + res.status);

      modalStatus.textContent = 'Puntaje enviado correctamente.';
      modalStatus.className = 'modal-status success';

      // Update local grid data
      updateLocalGrid(currentCell.season, currentCell.episode, selectedScore);

      // Reload comments
      loadComments(currentCell.season, currentCell.episode);

    } catch (e) {
      // If API unavailable, still update locally for demo
      modalStatus.textContent = 'Guardado localmente (API no disponible).';
      modalStatus.className = 'modal-status success';
      updateLocalGrid(currentCell.season, currentCell.episode, selectedScore);
    } finally {
      modalSubmit.disabled = false;
    }
  });

  function updateLocalGrid(seasonId, epNum, newScore) {
    var season = gridData.seasons.find(function (s) { return s.id === seasonId; });
    if (!season) return;

    var ep = season.episodes.find(function (e) { return e.episode === epNum; });
    if (ep) {
      // Recalculate avg with new vote
      var totalScore = ep.avg * ep.votes + newScore;
      ep.votes += 1;
      ep.avg = Math.round((totalScore / ep.votes) * 10) / 10;
    } else {
      season.episodes.push({ episode: epNum, avg: newScore, votes: 1 });
    }

    // Re-render grid
    renderGrid(gridData);

    // Update modal avg display
    var updatedEp = findEpisode(gridData, seasonId, epNum);
    if (updatedEp) {
      modalAvgValue.textContent = updatedEp.avg.toFixed(1);
      modalAvgVotes.textContent = '(' + updatedEp.votes + ' votos)';
    }
  }

  // ------ Load comments ------

  async function loadComments(season, episode) {
    commentsList.innerHTML = '<p class="no-comments">Cargando...</p>';

    try {
      var res = await fetch(API_BASE + '/api/ratings/' + season + '/' + episode);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var data = await res.json();

      if (data.ratings && data.ratings.length > 0) {
        commentsList.innerHTML = '';
        data.ratings.forEach(function (r) {
          var item = document.createElement('div');
          item.className = 'comment-item';
          var scoreColor = getScoreColor(r.score);
          var commentHtml = '<span class="comment-score" style="color:' + scoreColor + '">' + r.score + '/10</span>';
          if (r.comment) {
            commentHtml += '<span class="comment-text">' + escapeHtml(r.comment) + '</span>';
          }
          if (r.date) {
            commentHtml += '<span class="comment-date">' + r.date + '</span>';
          }
          item.innerHTML = commentHtml;
          commentsList.appendChild(item);
        });
      } else {
        commentsList.innerHTML = '<p class="no-comments">Todavia no hay comentarios para este programa.</p>';
      }
    } catch (e) {
      commentsList.innerHTML = '<p class="no-comments">No se pudieron cargar los comentarios.</p>';
    }
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ------ Init ------

  async function init() {
    var apiData = await fetchGridData();
    var data = apiData || generateSampleData();
    renderGrid(data);
  }

  init();
})();
