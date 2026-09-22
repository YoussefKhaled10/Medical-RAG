from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from frontend.api_client import APIClient

_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,50}$")


def _load_css() -> None:
    path = Path(__file__).resolve().parents[1] / "styles" / "auth_pages.css"
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _mark(page: str) -> None:
    st.markdown(f'<span class="rp-auth-marker rp-auth-{page}"></span>', unsafe_allow_html=True)


def _go(page: str) -> None:
    st.session_state.current_page = page
    st.rerun()


def _continue_as_guest(client: APIClient) -> None:
    """Open the main app without creating an authenticated session."""
    client.set_token("")
    st.session_state.auth_token = None
    st.session_state.refresh_token = None
    st.session_state.user = None
    st.session_state.user_info = None
    st.session_state.pending_registration = None
    st.session_state.pending_email = None
    _clear_chat_state()
    st.session_state.current_page = "app"
    st.rerun()


def _clear_chat_state() -> None:
    values = {
        "messages": [], "latest_response": None, "current_conv_id": None,
        "conversations": {}, "conversations_synced": False, "pending_question": None,
    }
    for key, value in values.items():
        st.session_state[key] = value


def _finish_auth(data: dict[str, Any], client: APIClient) -> None:
    token = str(data.get("access_token") or "")
    client.set_token(token)
    st.session_state.auth_token = token
    st.session_state.refresh_token = data.get("refresh_token")
    st.session_state.user = data.get("user") or {}
    st.session_state.user_info = st.session_state.user
    st.session_state.pending_registration = None
    st.session_state.pending_email = None
    _clear_chat_state()
    st.session_state.current_page = "app"
    st.rerun()


def _brand(page: str) -> None:
    title = {"login": "Welcome back", "register": "Create your account", "otp": "One step closer"}[page]
    description = {
        "login": "Securely access your private medical workspace and continue with trusted, evidence-based guidance.",
        "register": "Create a private RecoveryPath AI account for secure chats, document vaults, and high-confidence answers.",
        "otp": "Verify your email to activate your RecoveryPath AI workspace and keep your account secure.",
    }[page]
    st.markdown(
        f"""
        <section class="rp-brand-panel">
          <div class="rp-brand-shield-wrap" aria-hidden="true">
            <svg class="rp-brand-3d-shield" viewBox="0 0 240 280" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="120" cy="140" r="90" fill="url(#shieldGlow)" opacity="0.45" />
              <path d="M120 40 L190 80 L190 160 L120 200 L50 160 L50 80 Z" stroke="rgba(34,211,238,0.18)" stroke-width="1.5" stroke-dasharray="4 4" />
              <path d="M190 80 L230 60 M50 80 L10 60 M120 40 L120 10" stroke="rgba(34,211,238,0.14)" stroke-width="1.2" />
              <path d="M40 215 L120 250 L200 215 L120 180 Z" fill="#042335" stroke="#14b8a6" stroke-width="1.5" stroke-opacity="0.6"/>
              <path d="M40 215 L40 225 L120 260 L200 225 L200 215 L120 250 Z" fill="#021422" stroke="#0d9488" stroke-width="1.5" stroke-opacity="0.8"/>
              <path d="M55 205 L120 234 L185 205 L120 176 Z" fill="url(#pedestalTop)" stroke="#22d3ee" stroke-width="1.8"/>
              <path d="M55 205 L55 212 L120 241 L185 212 L185 205 L120 234 Z" fill="#032b3d" stroke="#14b8a6" stroke-width="1.5"/>
              <ellipse cx="120" cy="205" rx="45" ry="18" fill="url(#pedestalLight)" />
              <path d="M120 65 L175 95 L175 160 C175 195 120 220 120 220 C120 220 65 195 65 160 L65 95 Z" fill="url(#shieldGlass)" stroke="url(#shieldBorder)" stroke-width="2.5" filter="drop-shadow(0 0 16px rgba(20,184,166,0.55))"/>
              <path d="M120 75 L165 100 L165 155 C165 185 120 208 120 208 C120 208 75 185 75 155 L75 100 Z" stroke="rgba(167,243,208,0.35)" stroke-width="1.2" fill="none"/>
              <path d="M120 108 V156 M96 132 H144" stroke="#5eead4" stroke-width="9" stroke-linecap="round" filter="drop-shadow(0 0 10px #2dd4bf)"/>
              <defs>
                <radialGradient id="shieldGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.8"/>
                  <stop offset="100%" stop-color="#14b8a6" stop-opacity="0"/>
                </radialGradient>
                <linearGradient id="pedestalTop" x1="55" y1="176" x2="185" y2="234" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#083344"/>
                  <stop offset="50%" stop-color="#0e7490"/>
                  <stop offset="100%" stop-color="#083344"/>
                </linearGradient>
                <radialGradient id="pedestalLight" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0.7"/>
                  <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0"/>
                </radialGradient>
                <linearGradient id="shieldGlass" x1="65" y1="65" x2="175" y2="220" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.32"/>
                  <stop offset="40%" stop-color="#06b6d4" stop-opacity="0.18"/>
                  <stop offset="100%" stop-color="#042f2e" stop-opacity="0.45"/>
                </linearGradient>
                <linearGradient id="shieldBorder" x1="65" y1="65" x2="175" y2="220" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#5eead4"/>
                  <stop offset="50%" stop-color="#14b8a6"/>
                  <stop offset="100%" stop-color="#0d9488"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <header class="rp-brand-header">
            <div class="rp-brand-mark">
              <svg width="34" height="34" viewBox="0 0 32 32" fill="none">
                <path d="M16 3 L27 9.5 V22.5 L16 29 L5 22.5 V9.5 Z" stroke="#14b8a6" stroke-width="2.5" fill="rgba(20,184,166,0.15)"/>
                <path d="M16 10 V22 M10 16 H22" stroke="#2dd4bf" stroke-width="3" stroke-linecap="round"/>
              </svg>
            </div>
            <div class="rp-brand-wordmark">
              <div class="rp-brand-title">Recovery<span>PathAI</span></div>
              <small class="rp-brand-subtitle">Medical Knowledge Assistant</small>
            </div>
          </header>
          <div class="rp-brand-copy">
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
          <footer class="rp-brand-features">
            <div class="rp-feature-item">
              <svg class="rp-feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12l2 2 4-4"/>
              </svg>
              <div>
                <b>Secure &amp; Private</b>
                <small>Your account, chats, and documents stay isolated.</small>
              </div>
            </div>
            <div class="rp-feature-item">
              <svg class="rp-feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
              </svg>
              <div>
                <b>Evidence-Based</b>
                <small>Answers stay grounded in trusted sources.</small>
              </div>
            </div>
            <div class="rp-feature-item">
              <svg class="rp-feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
              <div>
                <b>Smart &amp; Fast</b>
                <small>Find relevant information without the noise.</small>
              </div>
            </div>
          </footer>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _columns(page: str):
    _load_css()
    _mark(page)
    return st.columns([1.02, 0.98], gap=None)


def render_login_page(client: APIClient) -> None:
    left, right = _columns("login")
    with left:
        _brand("login")
    with right:
        st.markdown('<div class="rp-kicker">WELCOME BACK</div><h1 class="rp-auth-title">Sign in</h1><p class="rp-auth-subtitle">Continue to your secure RecoveryPath AI workspace.</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            identity = st.text_input("Email or username", placeholder="Enter your email or username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Keep me signed in on this device")
            submitted = st.form_submit_button("Sign In  →", use_container_width=True)
        if submitted:
            if len(identity.strip()) < 3 or not password:
                st.error("Enter your email or username and password.")
            else:
                try:
                    with st.spinner("Signing in..."):
                        data = client.login(identity.strip(), password)
                    _finish_auth(data, client)
                except Exception as exc:
                    st.error(str(exc))
        st.markdown('<p class="rp-switch">New to RecoveryPath AI?</p>', unsafe_allow_html=True)
        if st.button("Create Account", key="login_to_register", use_container_width=True):
            _go("register")
        st.markdown('<div class="rp-auth-divider"><span>or</span></div>', unsafe_allow_html=True)
        if st.button("Continue as Guest", key="login_continue_guest", use_container_width=True):
            _continue_as_guest(client)


def render_register_page(client: APIClient) -> None:
    left, right = _columns("register")
    with left:
        _brand("register")
    with right:
        st.markdown('<div class="rp-kicker">GET STARTED</div><h1 class="rp-auth-title">Create your account</h1><p class="rp-auth-subtitle">Set up your secure medical workspace in just a few steps.</p>', unsafe_allow_html=True)
        with st.form("register_form"):
            full_name = st.text_input("Full name", placeholder="Enter your full name")
            username = st.text_input("Username", placeholder="Choose a unique username")
            email = st.text_input("Email address", placeholder="name@example.com")
            password = st.text_input("Password", type="password", placeholder="At least 8 characters")
            confirm = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
            submitted = st.form_submit_button("Create Account  →", use_container_width=True)
        if submitted:
            full_name, username, email = full_name.strip(), username.strip(), email.strip().lower()
            errors: list[str] = []
            if len(full_name) < 2: errors.append("Enter your full name.")
            if not _USERNAME.fullmatch(username): errors.append("Username must be 3-50 letters, numbers, dots, dashes, or underscores.")
            if not _EMAIL.fullmatch(email): errors.append("Enter a valid email address.")
            if len(password) < 8: errors.append("Password must contain at least 8 characters.")
            if password != confirm: errors.append("Passwords do not match.")
            if errors:
                for message in errors: st.error(message)
            else:
                try:
                    with st.spinner("Sending verification code..."):
                        client.send_otp(email)
                    st.session_state.pending_registration = {"full_name": full_name, "username": username, "email": email, "password": password}
                    st.session_state.pending_email = email
                    _go("otp")
                except Exception as exc:
                    st.error(str(exc))
        st.markdown('<p class="rp-switch rp-register-switch">Already have an account?</p>', unsafe_allow_html=True)
        if st.button("Sign In", key="register_to_login", use_container_width=True):
            _go("login")


def render_otp_page(client: APIClient) -> None:
    pending = st.session_state.get("pending_registration") or {}
    email = str(pending.get("email") or st.session_state.get("pending_email") or "")
    left, right = _columns("otp")
    with left:
        _brand("otp")
    with right:
        st.markdown(f'<div class="rp-otp-icon">✉</div><div class="rp-kicker">FINAL STEP</div><h1 class="rp-auth-title">Verify your account</h1><p class="rp-auth-subtitle">Enter the 6-digit code we sent to <b>{email or "your email"}</b>.</p>', unsafe_allow_html=True)
        with st.form("otp_form"):
            code = st.text_input("Verification code", max_chars=6, placeholder="000000")
            submitted = st.form_submit_button("Verify & Continue  →", use_container_width=True)
        if submitted:
            if not pending:
                st.error("Registration details expired. Create the account again.")
            elif not re.fullmatch(r"\d{6}", code.strip()):
                st.error("Enter the complete 6-digit verification code.")
            else:
                try:
                    with st.spinner("Verifying..."):
                        data = client.register(email=pending["email"], username=pending["username"], password=pending["password"], otp_code=code.strip(), full_name=pending.get("full_name"))
                    _finish_auth(data, client)
                except Exception as exc:
                    st.error(str(exc))
        if st.button("Resend code", key="otp_resend", use_container_width=True, disabled=not bool(email)):
            try:
                client.send_otp(email)
                st.success("A new verification code was sent.")
            except Exception as exc:
                st.error(str(exc))
        if st.button("← Back to Create Account", key="otp_back", use_container_width=True):
            _go("register")


def render_user_profile_sidebar(client: APIClient) -> None:
    user = st.session_state.get("user")
    if not isinstance(user, dict):
        if st.button("Sign in", use_container_width=True, key="sidebar_login"): _go("login")
        if st.button("Create account", use_container_width=True, key="sidebar_register"): _go("register")
        return
    # Show only the account username in the sidebar.
    username = escape(str(user.get("username") or "User"))
    st.markdown(
        f'<div style="margin:0.35rem 0 0.8rem;color:#f4ffff;font-size:1.05rem;font-weight:800;line-height:1.3;">{username}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign out", use_container_width=True, key="sidebar_logout"):
        try: client.logout(st.session_state.get("refresh_token"))
        except Exception: pass
        st.session_state.user = None
        st.session_state.user_info = None
        st.session_state.auth_token = None
        st.session_state.refresh_token = None
        _clear_chat_state()
        _go("login")
