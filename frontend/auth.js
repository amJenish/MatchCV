/** Shared auth helpers — include before page scripts. */
const API_BASE =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : window.location.origin;

function clearAuth() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

function redirectLogin() {
  clearAuth();
  window.location.replace("/login.html");
}

function getAccessToken() {
  return localStorage.getItem("access_token");
}

function authHeaders(extra) {
  const headers = { ...(extra || {}) };
  const token = getAccessToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/** OAuth callback may pass tokens in query or hash — store and strip from URL. */
function captureTokensFromUrl() {
  const hash = window.location.hash.startsWith("#")
    ? window.location.hash.slice(1)
    : "";
  const search = window.location.search.startsWith("?")
    ? window.location.search.slice(1)
    : "";
  const params = new URLSearchParams(hash || search);

  const access = params.get("access_token");
  const refresh = params.get("refresh_token");
  if (!access) return false;

  localStorage.setItem("access_token", access);
  if (refresh) {
    localStorage.setItem("refresh_token", refresh);
  }

  history.replaceState(null, "", window.location.pathname);
  return true;
}

/**
 * Returns the /auth/me payload if the user has a valid session, otherwise
 * redirects to login and returns null. Pages that only care about a
 * boolean can write `if (!(await requireAuth())) return;` — non-null is
 * truthy, null is falsy.
 */
async function requireAuth() {
  captureTokensFromUrl();
  const token = getAccessToken();
  if (!token) {
    redirectLogin();
    return null;
  }
  try {
    const res = await fetch(`${API_BASE}/auth/me`, { headers: authHeaders() });
    if (!res.ok) {
      redirectLogin();
      return null;
    }
    return await res.json();
  } catch {
    redirectLogin();
    return null;
  }
}

async function signOut() {
  const token = getAccessToken();
  if (token) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        headers: authHeaders(),
      });
    } catch {
      /* still clear local session */
    }
  }
  redirectLogin();
}

/**
 * Shared API helpers. Each one redirects to /login.html on 401 and
 * throws an Error whose message comes from the server's `error` field
 * (or a generic "Request failed (status)" fallback). The thrown error
 * carries `.status` so callers can branch on specific codes.
 */
async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (res.status === 401) { redirectLogin(); throw new Error("unauthorized"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) { redirectLogin(); throw new Error("unauthorized"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function apiPostMultipart(path, formData) {
  // Do NOT set Content-Type — the browser fills in the boundary itself.
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  if (res.status === 401) { redirectLogin(); throw new Error("unauthorized"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}
