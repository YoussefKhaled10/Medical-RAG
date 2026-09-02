"""RecoveryPath AI authentication UI for Streamlit.

Pixel-perfect implementation of the design mockup with wave headers,
teal gradient accents, step indicators, and bottom feature highlights.
Strictly no emojis and no forgot password link as requested.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import streamlit as st

from frontend.api_client import APIClient


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _asset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_auth_css() -> None:
    path = _asset_root() / "styles" / "auth.css"
    if path.exists():
        st.markdown(
            f"<style>{path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _logo_b64() -> str:
    path = _asset_root() / "assets" / "recoverypath_logo.svg"
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    return ""


def _go(page: str) -> None:
    st.session_state.current_page = page
    st.rerun()


def _render_top_header(active_step: int = 1) -> None:
    """Render the central RecoveryPath AI header and step navigation."""
    step1_class = "rp-step-tab active" if active_step == 1 else "rp-step-tab"
    step2_class = "rp-step-tab active" if active_step == 2 else "rp-step-tab"
    step3_class = "rp-step-tab active" if active_step == 3 else "rp-step-tab"

    st.markdown(
        f"""
        <div class="rp-main-header">
            <h1 class="rp-main-title">RecoveryPath AI</h1>
            <p class="rp-main-subtitle">Evidence-based recovery support</p>
        </div>
        <div class="rp-steps-nav">
            <div class="{step1_class}">1. Sign In</div>
            <div class="{step2_class}">2. Create Account</div>
            <div class="{step3_class}">3. Verify OTP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card_header() -> None:
    """Render the top wavy teal gradient banner and centered circular logo badge."""
    logo_src = _logo_b64()
    st.markdown(
        f"""
        <div class="rp-card-header-wave">
            <svg viewBox="0 0 500 80" preserveAspectRatio="none">
                <path d="M0,35 C150,85 350,-25 500,35 L500,80 L0,80 Z" fill="#ffffff"></path>
            </svg>
        </div>
        <div class="rp-card-logo-badge">
            <img src="{logo_src}" alt="RecoveryPath AI Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_floating_shield() -> None:
    """Render the bottom-right floating shield badge."""
    st.markdown(
        """
        <div class="rp-floating-shield">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bottom_features_bar() -> None:
    """Render the 4 feature pill cards below the active auth form."""
    st.markdown(
        """
        <div class="rp-bottom-features-row">
            <div class="rp-feature-pill">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <polyline points="9 12 11 14 15 10"/>
                </svg>
                <span>Trusted Evidence</span>
            </div>
            <div class="rp-feature-pill">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
                <span>Your Data is Protected</span>
            </div>
            <div class="rp-feature-pill">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
                <span>Simple & Fast Experience</span>
            </div>
            <div class="rp-feature-pill">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="9 12 11 14 15 10"/>
                </svg>
                <span>Built for Recovery</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _store_auth(data: dict[str, Any], client: APIClient) -> None:
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    if access_token:
        st.session_state.auth_token = str(access_token)
        client.set_token(str(access_token))
    if refresh_token:
        st.session_state.refresh_token = str(refresh_token)

    user = data.get("user") or data.get("profile")
    if isinstance(user, dict):
        st.session_state.user = user
    elif access_token:
        try:
            st.session_state.user = client.get_me()
        except Exception:
            st.session_state.user = None


def render_login_page(client: APIClient) -> None:
    """Render Screen 1: Sign In exactly matching the mockup."""
    _load_auth_css()
    _render_top_header(active_step=1)

    _, center_col, _ = st.columns([1.1, 1.4, 1.1])
    with center_col:
        _render_card_header()
        st.markdown(
            """
            <div class="rp-card-body-header">
                <h2>Welcome Back!</h2>
                <p>Sign in to access your account</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("rp_login_form", clear_on_submit=False):
            identity = st.text_input(
                "Email or Username",
                placeholder="Email or Username",
                label_visibility="visible",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Password",
                label_visibility="visible",
                autocomplete="current-password",
            )
            remember_me = st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not identity.strip() or not password:
                st.error("Please enter your email or username and password.")
            else:
                try:
                    with st.spinner("Signing in..."):
                        data = client.login(identity, password)
                    _store_auth(data, client)
                    st.session_state.current_page = "app"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown('<div class="rp-auth-divider-text">or</div>', unsafe_allow_html=True)
        if st.button("Don't have an account? Create account", use_container_width=True, key="login_to_register"):
            _go("register")

        _render_floating_shield()

    _render_bottom_features_bar()


def render_register_page(client: APIClient) -> None:
    """Render Screen 2: Create Account exactly matching the mockup."""
    _load_auth_css()
    _render_top_header(active_step=2)

    _, center_col, _ = st.columns([1.1, 1.4, 1.1])
    with center_col:
        _render_card_header()
        st.markdown(
            """
            <div class="rp-card-body-header">
                <h2>Create Account</h2>
                <p>Join RecoveryPath AI today</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pending = st.session_state.get("pending_registration") or {}
        with st.form("rp_register_form", clear_on_submit=False):
            full_name = st.text_input(
                "Full Name",
                value=str(pending.get("full_name") or ""),
                placeholder="Full Name",
                label_visibility="visible",
                autocomplete="name",
            )
            email = st.text_input(
                "Email",
                value=str(pending.get("email") or ""),
                placeholder="Email",
                label_visibility="visible",
                autocomplete="email",
            )
            username = st.text_input(
                "Username",
                value=str(pending.get("username") or ""),
                placeholder="Username",
                label_visibility="visible",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Password",
                label_visibility="visible",
                autocomplete="new-password",
            )
            confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm Password",
                label_visibility="visible",
                autocomplete="new-password",
            )
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            errors: list[str] = []
            if not _EMAIL_RE.match(email.strip()):
                errors.append("Please enter a valid email address.")
            if not _USERNAME_RE.match(username.strip()):
                errors.append("Username must be 3 to 32 characters using letters, numbers, dots or underscores.")
            if len(password) < 8:
                errors.append("Password must contain at least 8 characters.")
            if password != confirm:
                errors.append("Passwords do not match.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    with st.spinner("Sending verification code..."):
                        client.send_otp(email)
                    st.session_state.pending_registration = {
                        "full_name": full_name.strip() or None,
                        "email": email.strip(),
                        "username": username.strip(),
                        "password": password,
                    }
                    st.session_state.current_page = "otp"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        if st.button("Already have an account? Sign in", use_container_width=True, key="register_to_login"):
            _go("login")

        _render_floating_shield()

    _render_bottom_features_bar()


def render_otp_page(client: APIClient) -> None:
    """Render Screen 3: Verify OTP exactly matching the mockup."""
    _load_auth_css()
    pending = st.session_state.get("pending_registration") or {}
    email = str(pending.get("email") or "example@email.com")

    _render_top_header(active_step=3)

    _, center_col, _ = st.columns([1.1, 1.4, 1.1])
    with center_col:
        _render_card_header()
        st.markdown(
            f"""
            <div class="rp-card-body-header">
                <h2>Verify Your Email</h2>
                <p>We have sent a 6-digit code to</p>
                <p style="color: #0f766e; font-weight: 700; margin: 2px 0 6px 0;">{email}</p>
                <p style="font-size: 0.88rem; color: #64748b;">Please enter the code below</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("rp_otp_form", clear_on_submit=False):
            otp_code = st.text_input(
                "6-digit Code",
                max_chars=6,
                placeholder="1 2 3 4 5 6",
                label_visibility="collapsed",
                autocomplete="one-time-code",
            )
            st.markdown(
                """
                <div class="rp-otp-resend-text">
                    Didn't receive the code? <strong>Resend code in 00:45</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Verify", use_container_width=True)

        if submitted:
            code = re.sub(r"\D", "", otp_code)
            if len(code) != 6:
                st.error("Please enter the complete 6-digit verification code.")
            else:
                try:
                    with st.spinner("Verifying code..."):
                        data = client.register(
                            email=email,
                            username=str(pending.get("username") or ""),
                            password=str(pending.get("password") or ""),
                            otp_code=code,
                            full_name=pending.get("full_name"),
                        )
                    _store_auth(data, client)
                    st.session_state.pending_registration = None
                    st.session_state.current_page = "app"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown('<div class="rp-outline-btn">', unsafe_allow_html=True)
        if st.button("Change Email", use_container_width=True, key="otp_change_btn"):
            _go("register")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Back to Sign In", use_container_width=True, key="otp_to_login"):
            _go("login")

        _render_floating_shield()

    _render_bottom_features_bar()


def render_user_profile_sidebar(client: APIClient) -> None:
    """Render top buttons and user profile in the dark navy sidebar."""
    user = st.session_state.get("user")
    token = st.session_state.get("auth_token")

    if not user and token:
        try:
            client.set_token(str(token))
            user = client.get_me()
            st.session_state.user = user
        except Exception:
            user = None

    if not isinstance(user, dict):
        st.markdown('<div class="rp-sidebar-actions">', unsafe_allow_html=True)
        st.markdown('<div class="rp-sidebar-btn-white">', unsafe_allow_html=True)
        if st.button("Sign in", use_container_width=True, key="sidebar_sign_in"):
            _go("login")
        if st.button("Create account", use_container_width=True, key="sidebar_create_account"):
            _go("register")
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    name = str(user.get("full_name") or user.get("username") or "User")
    detail = str(user.get("email") or user.get("username") or "Verified Account")
    initial = name[:1].upper() if name else "U"
    vault_id = user.get("private_project_id") or 1

    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(13,148,136,0.3); border-radius: 16px; padding: 12px 14px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #0d9488, #06b6d4); display: flex; align-items: center; justify-content: center; font-weight: 700; color: white;">
                    {initial}
                </div>
                <div>
                    <strong style="color: #ffffff; font-size: 0.92rem; display: block;">{name}</strong>
                    <span style="color: #2dd4bf; font-size: 0.76rem; font-weight: 600;">Private Vault #{vault_id}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", use_container_width=True, key="sidebar_logout"):
        client.logout(st.session_state.get("refresh_token"))
        st.session_state.user = None
        st.session_state.auth_token = None
        st.session_state.refresh_token = None
        st.session_state.current_page = "login"
        st.rerun()
