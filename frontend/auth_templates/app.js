"use strict";

const API_BASE_URL = window.RECOVERYPATH_API_URL || "http://127.0.0.1:8000";

function showStatus(message, type = "error") {
  const box = document.querySelector("#statusMessage");
  if (!box) return;
  box.textContent = message;
  box.className = `status-message show ${type}`;
}

function clearErrors(form) {
  form.querySelectorAll(".form-group").forEach(group => group.classList.remove("has-error"));
  form.querySelectorAll(".field").forEach(field => field.classList.remove("invalid"));
  const status = document.querySelector("#statusMessage");
  if (status) status.className = "status-message";
}

function markError(input, message) {
  const group = input.closest(".form-group");
  const field = input.closest(".field");
  if (group) {
    group.classList.add("has-error");
    const error = group.querySelector(".error-text");
    if (error) error.textContent = message;
  }
  if (field) field.classList.add("invalid");
}

function setLoading(button, loading, text) {
  if (!button) return;
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent;
  button.disabled = loading;
  button.textContent = loading ? text : button.dataset.originalText;
}

function setupPasswordToggles() {
  document.querySelectorAll("[data-password-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const input = document.querySelector(button.dataset.passwordToggle);
      if (!input) return;
      const visible = input.type === "text";
      input.type = visible ? "password" : "text";
      button.textContent = visible ? "◉" : "◌";
      button.setAttribute("aria-label", visible ? "Show password" : "Hide password");
    });
  });
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  let data = {};
  try { data = await response.json(); } catch (_) {}
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request failed. Please try again.";
    throw new Error(detail);
  }
  return data;
}

function setupLogin() {
  const form = document.querySelector("#loginForm");
  if (!form) return;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    clearErrors(form);
    const identity = form.email_or_username;
    const password = form.password;
    let valid = true;
    if (identity.value.trim().length < 3) { markError(identity, "Enter your email or username."); valid = false; }
    if (!password.value) { markError(password, "Enter your password."); valid = false; }
    if (!valid) return;
    const submit = form.querySelector("button[type=submit]");
    setLoading(submit, true, "Signing in...");
    try {
      const data = await apiRequest("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ email_or_username: identity.value.trim(), password: password.value })
      });
      localStorage.setItem("access_token", data.access_token || "");
      localStorage.setItem("refresh_token", data.refresh_token || "");
      localStorage.setItem("recoverypath_user", JSON.stringify(data.user || {}));
      showStatus("Signed in successfully. Redirecting...", "success");
      setTimeout(() => { window.location.href = "../index.html"; }, 700);
    } catch (error) { showStatus(error.message); }
    finally { setLoading(submit, false, "Signing in..."); }
  });
}

function setupRegister() {
  const form = document.querySelector("#registerForm");
  if (!form) return;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    clearErrors(form);
    const data = Object.fromEntries(new FormData(form));
    let valid = true;
    if (data.full_name.trim().length < 2) { markError(form.full_name, "Enter your full name."); valid = false; }
    if (!/^[A-Za-z0-9_.-]{3,50}$/.test(data.username)) { markError(form.username, "Use 3-50 letters, numbers, dots, dashes or underscores."); valid = false; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) { markError(form.email, "Enter a valid email address."); valid = false; }
    if (data.password.length < 8) { markError(form.password, "Password must contain at least 8 characters."); valid = false; }
    if (data.password !== data.confirm_password) { markError(form.confirm_password, "Passwords do not match."); valid = false; }
    if (!valid) return;
    const submit = form.querySelector("button[type=submit]");
    setLoading(submit, true, "Sending code...");
    try {
      await apiRequest("/api/v1/auth/send-otp", { method: "POST", body: JSON.stringify({ email: data.email.trim() }) });
      sessionStorage.setItem("pending_registration", JSON.stringify({
        full_name: data.full_name.trim(), username: data.username.trim(), email: data.email.trim(), password: data.password
      }));
      window.location.href = `otp.html?email=${encodeURIComponent(data.email.trim())}`;
    } catch (error) { showStatus(error.message); }
    finally { setLoading(submit, false, "Sending code..."); }
  });
}

function setupOtp() {
  const form = document.querySelector("#otpForm");
  if (!form) return;
  const inputs = [...document.querySelectorAll(".otp-input")];
  const email = new URLSearchParams(location.search).get("email") || "your email";
  const emailNode = document.querySelector("#otpEmail");
  if (emailNode) emailNode.textContent = email;

  inputs.forEach((input, index) => {
    input.addEventListener("input", () => {
      input.value = input.value.replace(/\D/g, "").slice(0, 1);
      if (input.value && inputs[index + 1]) inputs[index + 1].focus();
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Backspace" && !input.value && inputs[index - 1]) inputs[index - 1].focus();
    });
    input.addEventListener("paste", event => {
      event.preventDefault();
      const digits = event.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
      digits.split("").forEach((digit, i) => { if (inputs[i]) inputs[i].value = digit; });
      if (inputs[Math.min(digits.length, 5)]) inputs[Math.min(digits.length, 5)].focus();
    });
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const otp = inputs.map(input => input.value).join("");
    const pending = JSON.parse(sessionStorage.getItem("pending_registration") || "null");
    if (!/^\d{6}$/.test(otp)) { showStatus("Enter the complete 6-digit code."); return; }
    if (!pending) { showStatus("Registration details expired. Please create your account again."); return; }
    const submit = form.querySelector("button[type=submit]");
    setLoading(submit, true, "Verifying...");
    try {
      const result = await apiRequest("/api/v1/auth/register", {
        method: "POST",
        body: JSON.stringify({ ...pending, otp_code: otp })
      });
      localStorage.setItem("access_token", result.access_token || "");
      localStorage.setItem("refresh_token", result.refresh_token || "");
      localStorage.setItem("recoverypath_user", JSON.stringify(result.user || {}));
      sessionStorage.removeItem("pending_registration");
      showStatus("Account verified successfully. Redirecting...", "success");
      setTimeout(() => { window.location.href = "../index.html"; }, 700);
    } catch (error) { showStatus(error.message); }
    finally { setLoading(submit, false, "Verifying..."); }
  });

  const resendButton = document.querySelector("#resendButton");
  const timerNode = document.querySelector("#resendTimer");
  let seconds = 60;
  const tick = () => {
    if (!resendButton || !timerNode) return;
    timerNode.textContent = `(${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")})`;
    resendButton.disabled = seconds > 0;
    if (seconds > 0) { seconds -= 1; setTimeout(tick, 1000); }
    else timerNode.textContent = "";
  };
  tick();
  resendButton?.addEventListener("click", async () => {
    try {
      await apiRequest("/api/v1/auth/send-otp", { method: "POST", body: JSON.stringify({ email }) });
      seconds = 60; tick(); showStatus("A new verification code was sent.", "success");
    } catch (error) { showStatus(error.message); }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupPasswordToggles();
  setupLogin();
  setupRegister();
  setupOtp();
});
