(function () {
    const POLL_INTERVAL_MS = 5 * 60 * 1000;
    const state = {
        currentFlare: null,
        ready: false,
        elements: {},
        pollTimer: null,
    };

    function isFlareAlertDisplayDisabled() {
        return document.body && document.body.dataset.flareAlertEnabled === 'false';
    }

    function formatUtcTimestamp(value) {
        if (!value) {
            return 'Unknown time';
        }

        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return 'Unknown time';
        }

        const month = date.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
        const day = date.toLocaleString('en-US', { day: 'numeric', timeZone: 'UTC' });
        const hours = String(date.getUTCHours()).padStart(2, '0');
        const minutes = String(date.getUTCMinutes()).padStart(2, '0');
        return `${month} ${day}, ${hours}:${minutes} UTC`;
    }

    function buildDismissKey(flareId) {
        return `wtus_flare_alert_dismissed_${flareId}`;
    }

    function isDismissed(flareId) {
        if (!flareId) {
            return false;
        }
        return window.localStorage.getItem(buildDismissKey(flareId)) === '1';
    }

    function dismissCurrentFlare() {
        if (!state.currentFlare || !state.currentFlare.id) {
            return;
        }

        window.localStorage.setItem(buildDismissKey(state.currentFlare.id), '1');
        hidePopup();
        closeModal();
    }

    function ensureUi() {
        if (state.ready || !document.body) {
            return;
        }

        const popup = document.createElement('aside');
        popup.className = 'flare-alert-popup';
        popup.hidden = true;
        popup.setAttribute('aria-live', 'polite');
        popup.setAttribute('role', 'button');
        popup.setAttribute('tabindex', '0');
        popup.innerHTML = [
            '<div class="flare-alert-popup__inner">',
            '  <div class="flare-alert-popup__top">',
            '    <div>',
            '      <div class="flare-alert-popup__eyebrow">Solar Flare Alert</div>',
            '      <h3 class="flare-alert-popup__title"></h3>',
            '      <div class="flare-alert-popup__meta"></div>',
            '    </div>',
            '    <div style="display:flex; gap:0.55rem; align-items:flex-start;">',
            '      <div class="flare-alert-popup__badge"></div>',
            '      <button class="flare-alert-popup__close" type="button" aria-label="Dismiss flare alert">&times;</button>',
            '    </div>',
            '  </div>',
            '  <div class="flare-alert-popup__actions">',
            '    <span class="flare-alert-popup__hint">Click for event graphic</span>',
            '    <span class="flare-alert-popup__hint flare-alert-popup__rscale"></span>',
            '  </div>',
            '</div>'
        ].join('');

        const modal = document.createElement('div');
        modal.className = 'flare-alert-modal';
        modal.hidden = true;
        modal.innerHTML = [
            '<div class="flare-alert-modal__dialog" role="dialog" aria-modal="true" aria-labelledby="flare-alert-modal-title">',
            '  <div class="flare-alert-modal__header">',
            '    <div>',
            '      <div class="flare-alert-modal__kicker">Latest Flare Event</div>',
            '      <h3 id="flare-alert-modal-title" class="flare-alert-modal__title"></h3>',
            '      <div class="flare-alert-modal__meta"></div>',
            '    </div>',
            '    <button class="flare-alert-modal__close" type="button" aria-label="Close flare alert graphic">&times;</button>',
            '  </div>',
            '  <div class="flare-alert-modal__body">',
            '    <div class="flare-alert-modal__figure-shell">',
            '      <img class="flare-alert-modal__image" alt="Latest solar flare event graphic">',
            '      <div class="flare-alert-modal__loading">Loading flare graphic...</div>',
            '      <div class="flare-alert-modal__error" hidden>Unable to load the flare graphic right now.</div>',
            '    </div>',
            '    <div class="flare-alert-modal__footer">',
            '      <span>Graphic uses live GOES and SDO imagery with current SWPC chart data.</span>',
            '      <button class="flare-alert-modal__dismiss" type="button">Dismiss this alert</button>',
            '    </div>',
            '  </div>',
            '</div>'
        ].join('');

        document.body.appendChild(popup);
        document.body.appendChild(modal);

        state.elements = {
            popup,
            popupTitle: popup.querySelector('.flare-alert-popup__title'),
            popupMeta: popup.querySelector('.flare-alert-popup__meta'),
            popupBadge: popup.querySelector('.flare-alert-popup__badge'),
            popupRScale: popup.querySelector('.flare-alert-popup__rscale'),
            popupClose: popup.querySelector('.flare-alert-popup__close'),
            modal,
            modalTitle: modal.querySelector('.flare-alert-modal__title'),
            modalMeta: modal.querySelector('.flare-alert-modal__meta'),
            modalClose: modal.querySelector('.flare-alert-modal__close'),
            modalDismiss: modal.querySelector('.flare-alert-modal__dismiss'),
            modalImage: modal.querySelector('.flare-alert-modal__image'),
            modalLoading: modal.querySelector('.flare-alert-modal__loading'),
            modalError: modal.querySelector('.flare-alert-modal__error'),
        };

        popup.addEventListener('click', function () {
            openModal();
        });
        popup.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                openModal();
            }
        });
        state.elements.popupClose.addEventListener('click', function (event) {
            event.stopPropagation();
            dismissCurrentFlare();
        });

        modal.addEventListener('click', function (event) {
            if (event.target === modal) {
                closeModal();
            }
        });
        state.elements.modalClose.addEventListener('click', closeModal);
        state.elements.modalDismiss.addEventListener('click', dismissCurrentFlare);

        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });

        state.ready = true;
    }

    function applyAccent(flare) {
        ensureUi();
        const accent = flare.base_color || flare.color || '#facc15';
        state.elements.popup.style.setProperty('--flare-alert-accent', accent);
        state.elements.modal.style.setProperty('--flare-alert-accent', accent);
    }

    function hidePopup() {
        ensureUi();
        state.elements.popup.hidden = true;
    }

    function renderPopup(flare) {
        ensureUi();
        applyAccent(flare);

        const sourceBits = [];
        if (flare.source_region) {
            sourceBits.push(flare.source_region);
        }
        if (flare.location) {
            sourceBits.push(flare.location);
        }

        state.elements.popupTitle.textContent = `${flare.event_class} flare detected`;
        state.elements.popupMeta.textContent = `${formatUtcTimestamp(flare.peak_time || flare.time)}${sourceBits.length ? ` • ${sourceBits.join(' • ')}` : ''}`;
        state.elements.popupBadge.textContent = flare.event_class || '--';
        state.elements.popupRScale.textContent = flare.r_scale_label ? `Peak ${flare.r_scale_label}` : '';
        state.elements.popup.hidden = false;
    }

    function buildGraphicUrl(flare) {
        if (flare.graphic_url) {
            const separator = flare.graphic_url.indexOf('?') >= 0 ? '&' : '?';
            return `${flare.graphic_url}${separator}event_id=${encodeURIComponent(flare.id || Date.now())}`;
        }

        const params = new URLSearchParams();
        if (flare.event_class) params.set('event_class', flare.event_class);
        if (flare.time) params.set('start_time', flare.time);
        if (flare.peak_time) params.set('peak_time', flare.peak_time);
        if (flare.end_time) params.set('end_time', flare.end_time);
        if (flare.region_number) params.set('region', flare.region_number);
        if (flare.location) params.set('location', flare.location);
        if (flare.source_region) params.set('source_region', flare.source_region);
        if (flare.r_scale_label) params.set('r_scale', flare.r_scale_label);
        if (flare.id) params.set('event_id', flare.id);
        return `/api/flare-alert-graphic?${params.toString()}`;
    }

    function openModal() {
        ensureUi();
        if (!state.currentFlare) {
            return;
        }

        applyAccent(state.currentFlare);
        state.elements.modalTitle.textContent = `${state.currentFlare.event_class} flare event graphic`;

        const metaParts = [formatUtcTimestamp(state.currentFlare.peak_time || state.currentFlare.time)];
        if (state.currentFlare.source_region) {
            metaParts.push(state.currentFlare.source_region);
        }
        if (state.currentFlare.location) {
            metaParts.push(state.currentFlare.location);
        }
        state.elements.modalMeta.textContent = metaParts.join(' • ');

        state.elements.modal.hidden = false;
        document.body.classList.add('flare-alert-modal-open');
        state.elements.modalLoading.hidden = false;
        state.elements.modalError.hidden = true;
        state.elements.modalImage.removeAttribute('src');

        const nextUrl = buildGraphicUrl(state.currentFlare) + `&t=${encodeURIComponent(Date.now())}`;
        state.elements.modalImage.onload = function () {
            state.elements.modalLoading.hidden = true;
            state.elements.modalError.hidden = true;
        };
        state.elements.modalImage.onerror = function () {
            state.elements.modalLoading.hidden = true;
            state.elements.modalError.hidden = false;
        };
        state.elements.modalImage.src = nextUrl;
    }

    function closeModal() {
        if (!state.ready) {
            return;
        }
        state.elements.modal.hidden = true;
        document.body.classList.remove('flare-alert-modal-open');
        state.elements.modalLoading.hidden = true;
        state.elements.modalError.hidden = true;
    }

    async function pollFlareAlert() {
        if (isFlareAlertDisplayDisabled()) {
            state.currentFlare = null;
            if (state.ready) {
                hidePopup();
                closeModal();
            }
            return;
        }

        ensureUi();

        try {
            const response = await fetch(`/api/flare-alert?t=${Date.now()}`, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`Flare alert request failed with status ${response.status}`);
            }

            const payload = await response.json();
            const flare = payload && payload.active ? payload.flare : null;

            if (!flare || !flare.id) {
                state.currentFlare = null;
                hidePopup();
                return;
            }

            state.currentFlare = flare;
            if (isDismissed(flare.id)) {
                hidePopup();
                return;
            }

            renderPopup(flare);
        } catch (error) {
            console.warn('Unable to refresh flare alert.', error);
        }
    }

    function init() {
        if (isFlareAlertDisplayDisabled()) {
            if (state.pollTimer) {
                window.clearInterval(state.pollTimer);
                state.pollTimer = null;
            }
            return;
        }

        ensureUi();
        pollFlareAlert();
        if (state.pollTimer) {
            window.clearInterval(state.pollTimer);
        }
        state.pollTimer = window.setInterval(pollFlareAlert, POLL_INTERVAL_MS);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }

    window.WTUSFlareAlert = {
        refresh: pollFlareAlert,
        dismiss: dismissCurrentFlare,
    };
})();