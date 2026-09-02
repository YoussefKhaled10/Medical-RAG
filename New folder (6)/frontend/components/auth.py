"""RecoveryPath AI authentication UI for Streamlit.

Provides login, registration, OTP verification, and authenticated profile controls.
The module uses the existing APIClient methods without requiring forgot-password routes.
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
    if not path.exists():
        return
    st.markdown(
        f"<style>{path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _logo_html() -> str:
    path = _asset_root() / "assets" / "recoverypath_logo.svg"
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return (
            '<img class="rp-auth-logo-img" '
            f'src="data:image/svg+xml;base64,{encoded}" alt="RecoveryPath AI logo">'
        )
    return '<span class="rp-auth-logo-fallback">RP</span>'


def _go(page: str) -> None:
    st.session_state.current_page = page
    st.rerun()


def _hero(title: str, description: str, eyebrow: str, step: int) -> None:
    labels = ("Account", "Verify", "Ready")
    steps = []
    for index, label in enumerate(labels, start=1):
        active = " active" if index <= step else ""
        steps.append(
            f'<div class="rp-step{active}"><span class="rp-step-num">{index}</span>'
            f'<span>{label}</span></div>'
        )
        if index != len(labels):
            steps.append('<span class="rp-step-divider"></span>')

    st.markdown(
        f"""
        <div class="rp-auth-page"></div>
        <section class="rp-auth-hero">
          <div class="rp-auth-grid"></div>
          <div class="rp-auth-glow rp-glow-a"></div>
          <div class="rp-auth-glow rp-glow-b"></div>
          <div class="rp-brand-lockup">
            <div class="rp-brand-icon">{_logo_html()}</div>
            <div class="rp-brand-name">RecoveryPath AI</div>
          </div>
          <span class="rp-auth-eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{description}</p>
          <div class="rp-hero-pills">
            <span class="rp-hero-pill"><i class="rp-hero-pill-dot"></i>Private account</span>
            <span class="rp-hero-pill"><i class="rp-hero-pill-dot"></i>Evidence-based support</span>
            <span class="rp-hero-pill"><i class="rp-hero-pill-dot"></i>Secure verification</span>
          </div>
        </section>
        <div class="rp-step-indicator">{''.join(steps)}</div>
        """,
        unsafe_allow_html=True,
    )


def _card_heading(title: str, description: str) -> None:
    st.markdown(
        f'<div class="rp-card-heading"><h2>{title}</h2><p>{description}</p></div>',
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
    _load_auth_css()
    _hero(
        "Welcome back",
        "Sign in to access your account and continue with evidence-based recovery support.",
        "SECURE SIGN IN",
        1,
    )

    _, center, _ = st.columns([1.1, 1.35, 1.1])
    with center:
        _card_heading("Welcome Back!", "Sign in to access your account.")
        with st.form("rp_login_form", clear_on_submit=False):
            identity = st.text_input(
                "Email or username",
                placeholder="you@example.com or username",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button("Sign in securely", use_container_width=True)

        if submitted:
            if not identity.strip() or not password:
                st.error("Enter your email or username and password.")
            else:
                try:
                    with st.spinner("Signing you in..."):
                        data = client.login(identity, password)
                    _store_auth(data, client)
                    st.session_state.current_page = "app"
                    st.success("Signed in successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown('<div class="rp-form-divider"></div>', unsafe_allow_html=True)
        st.caption("Don’t have an account?")
        if st.button("Create Account", use_container_width=True, key="login_to_register"):
            _go("register")
        st.markdown(
            '<div class="rp-trust-row"><span>Encrypted connection</span>'
            '<span>Private session</span><span>No medical diagnosis</span></div>',
            unsafe_allow_html=True,
        )


def render_register_page(client: APIClient) -> None:
    _load_auth_css()
    _hero(
        "Create your account",
        "Join RecoveryPath AI today and verify your email with a secure one-time code.",
        "CREATE ACCOUNT",
        1,
    )

    _, center, _ = st.columns([1.0, 1.55, 1.0])
    with center:
        _card_heading("Create Account", "Enter your details to begin.")
        pending = st.session_state.get("pending_registration") or {}
        with st.form("rp_register_form", clear_on_submit=False):
            full_name = st.text_input(
                "Full name (optional)",
                value=str(pending.get("full_name") or ""),
                placeholder="Your full name",
                autocomplete="name",
            )
            email = st.text_input(
                "Email address",
                value=str(pending.get("email") or ""),
                placeholder="you@example.com",
                autocomplete="email",
            )
            username = st.text_input(
                "Username",
                value=str(pending.get("username") or ""),
                placeholder="3 to 32 letters, numbers, dots or underscores",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 8 characters",
                autocomplete="new-password",
            )
            confirm = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Enter the password again",
                autocomplete="new-password",
            )
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            errors: list[str] = []
            if not _EMAIL_RE.match(email.strip()):
                errors.append("Enter a valid email address.")
            if not _USERNAME_RE.match(username.strip()):
                errors.append("Username must be 3 to 32 characters using letters, numbers, dots, hyphens or underscores.")
            if len(password) < 8:
                errors.append("Password must contain at least 8 characters.")
            if password != confirm:
                errors.append("Passwords do not match.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    with st.spinner("Sending your verification code..."):
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

        st.markdown('<div class="rp-form-divider"></div>', unsafe_allow_html=True)
        if st.button("Already have an account? Sign in", use_container_width=True, key="register_to_login"):
            _go("login")


def render_otp_page(client: APIClient) -> None:
    _load_auth_css()
    pending = st.session_state.get("pending_registration") or {}
    email = str(pending.get("email") or "")
    if not email:
        st.warning("Start registration before verifying an email address.")
        if st.button("Go to registration", use_container_width=True):
            _go("register")
        return

    _hero(
        "Verify your email",
        "We sent a six-digit code to your email address.",
        "EMAIL VERIFICATION",
        2,
    )

    _, center, _ = st.columns([1.1, 1.35, 1.1])
    with center:
        _card_heading("Verify Your Email", "Enter the six-digit code below.")
        st.markdown(
            f'<div class="rp-email-chip"><span>Code sent to</span><strong>{email}</strong></div>',
            unsafe_allow_html=True,
        )
        with st.form("rp_otp_form", clear_on_submit=False):
            otp_code = st.text_input(
                "One-time code",
                max_chars=6,
                placeholder="000000",
                autocomplete="one-time-code",
            )
            submitted = st.form_submit_button("Verify", use_container_width=True)

        if submitted:
            code = re.sub(r"\D", "", otp_code)
            if len(code) != 6:
                st.error("Enter the complete six-digit code.")
            else:
                try:
                    with st.spinner("Verifying your account..."):
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
                    st.success("Your account is ready.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        resend_col, change_col = st.columns(2)
        with resend_col:
            if st.button("Resend code", use_container_width=True, key="otp_resend"):
                try:
                    with st.spinner("Sending a new code..."):
                        client.send_otp(email)
                    st.success("A new verification code was sent.")
                except Exception as exc:
                    st.error(str(exc))
        with change_col:
            if st.button("Change Email", use_container_width=True, key="otp_change"):
                _go("register")


def render_user_profile_sidebar(client: APIClient) -> None:
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
        login_col, register_col = st.columns(2)
        with login_col:
            if st.button("Sign in", use_container_width=True, key="sidebar_login"):
                _go("login")
        with register_col:
            if st.button("Create account", use_container_width=True, key="sidebar_register"):
                _go("register")
        return

    name = str(user.get("full_name") or user.get("username") or "RecoveryPath user")
    detail = str(user.get("email") or user.get("username") or "Signed in")
    initial = name[:1].upper() if name else "R"
    st.markdown(
        f'<div class="rp-profile-card"><div class="rp-profile-avatar">{initial}</div>'
        f'<div class="rp-profile-copy"><strong>{name}</strong><span>{detail}</span></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign out", use_container_width=True, key="sidebar_logout"):
        client.logout(st.session_state.get("refresh_token"))
        st.session_state.user = None
        st.session_state.auth_token = None
        st.session_state.refresh_token = None
        st.session_state.current_page = "app"
        st.rerun()
