// === Paren la Mano - Archive Site JS ===

document.addEventListener('DOMContentLoaded', () => {
  // Mobile nav toggle
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });

    // Close menu on link click
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
      });
    });
  }

  // Episode loading (for episodes.html and index.html)
  const episodesContainer = document.getElementById('episodes-container');
  const latestContainer = document.getElementById('latest-episodes');

  if (episodesContainer || latestContainer) {
    loadEpisodes();
  }
});

// === Episode loading and rendering ===

let allEpisodes = [];

async function loadEpisodes() {
  try {
    const response = await fetch('data/episodes.json');
    if (!response.ok) throw new Error('No se pudieron cargar los episodios');
    allEpisodes = await response.json();

    // Sort by date descending
    allEpisodes.sort((a, b) => new Date(b.fecha) - new Date(a.fecha));

    // Render on episodes page
    const episodesContainer = document.getElementById('episodes-container');
    if (episodesContainer) {
      renderEpisodes(allEpisodes, episodesContainer);
      updateResultsCount(allEpisodes.length);
      setupSearch();
    }

    // Render latest on index page
    const latestContainer = document.getElementById('latest-episodes');
    if (latestContainer) {
      renderEpisodes(allEpisodes.slice(0, 3), latestContainer);
    }

  } catch (error) {
    console.error('Error cargando episodios:', error);
    const container = document.getElementById('episodes-container') || document.getElementById('latest-episodes');
    if (container) {
      container.innerHTML = `
        <div class="no-results">
          <h3>Error al cargar los episodios</h3>
          <p>Intenta recargar la pagina.</p>
        </div>
      `;
    }
  }
}

function renderEpisodes(episodes, container) {
  if (episodes.length === 0) {
    container.innerHTML = `
      <div class="no-results">
        <h3>No se encontraron episodios</h3>
        <p>Intenta con otra busqueda o cambia los filtros.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = episodes.map(ep => createEpisodeCard(ep)).join('');
}

function createEpisodeCard(ep) {
  const formattedDate = formatDate(ep.fecha);
  const youtubeUrl = `https://www.youtube.com/watch?v=${ep.youtube_id}`;
  const thumbUrl = `https://img.youtube.com/vi/${ep.youtube_id}/hqdefault.jpg`;

  return `
    <article class="episode-card">
      <div class="episode-thumb">
        <a href="${youtubeUrl}" target="_blank" rel="noopener">
          <img src="${thumbUrl}" alt="${ep.titulo}" loading="lazy">
        </a>
        <span class="duration-badge">${ep.duracion}</span>
      </div>
      <div class="episode-body">
        <div class="episode-meta">
          <span>${formattedDate}</span>
          <span>${ep.duracion}</span>
        </div>
        <h3><a href="${youtubeUrl}" target="_blank" rel="noopener">${ep.titulo}</a></h3>
        <p>${ep.descripcion}</p>
        <div class="episode-actions">
          <a href="${youtubeUrl}" target="_blank" rel="noopener">Ver en YouTube</a>
        </div>
      </div>
    </article>
  `;
}

function formatDate(dateStr) {
  const date = new Date(dateStr + 'T00:00:00');
  const months = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
  ];
  return `${date.getDate()} de ${months[date.getMonth()]} ${date.getFullYear()}`;
}

// === Search & Filter ===

function setupSearch() {
  const searchInput = document.getElementById('search-input');
  const sortSelect = document.getElementById('sort-select');

  if (searchInput) {
    searchInput.addEventListener('input', debounce(applyFilters, 300));
  }

  if (sortSelect) {
    sortSelect.addEventListener('change', applyFilters);
  }
}

function applyFilters() {
  const searchInput = document.getElementById('search-input');
  const sortSelect = document.getElementById('sort-select');
  const container = document.getElementById('episodes-container');

  if (!container) return;

  const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
  const sort = sortSelect ? sortSelect.value : 'newest';

  let filtered = allEpisodes.filter(ep => {
    if (!query) return true;
    return (
      ep.titulo.toLowerCase().includes(query) ||
      ep.descripcion.toLowerCase().includes(query) ||
      ep.numero.toString().includes(query)
    );
  });

  // Sort
  switch (sort) {
    case 'oldest':
      filtered.sort((a, b) => new Date(a.fecha) - new Date(b.fecha));
      break;
    case 'newest':
    default:
      filtered.sort((a, b) => new Date(b.fecha) - new Date(a.fecha));
      break;
    case 'number-asc':
      filtered.sort((a, b) => a.numero - b.numero);
      break;
    case 'number-desc':
      filtered.sort((a, b) => b.numero - a.numero);
      break;
  }

  renderEpisodes(filtered, container);
  updateResultsCount(filtered.length);
}

function updateResultsCount(count) {
  const el = document.getElementById('results-count');
  if (el) {
    el.textContent = count === 1
      ? '1 episodio encontrado'
      : `${count} episodios encontrados`;
  }
}

function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}
