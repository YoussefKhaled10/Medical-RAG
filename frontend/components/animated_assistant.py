import streamlit as st


def render_floating_assistant() -> None:
    """Render a persistent animated assistant with a CSS-only close control."""
    st.markdown(
        """
        <div class="rp-floating-assistant" aria-live="polite">
          <input
            class="rp-message-toggle"
            id="rp-message-toggle"
            type="checkbox"
            aria-label="Hide welcome message"
          >
          <div class="rp-floating-message">
            <label
              class="rp-message-close"
              for="rp-message-toggle"
              title="Hide message"
              aria-label="Hide welcome message"
            >×</label>
            <strong>Hello! I'm RecoveryPath AI</strong>
            <span>Ask me anything about alcohol recovery.</span>
          </div>
          <div class="rp-floating-bot" aria-hidden="true">
            <span class="rp-bot-spark rp-spark-a"></span>
            <span class="rp-bot-spark rp-spark-b"></span>
            <span class="rp-bot-spark rp-spark-c"></span>
            <div class="rp-bot-glow"></div>
            <div class="rp-bot-antenna"><i></i></div>
            <div class="rp-bot-ear rp-ear-left"></div>
            <div class="rp-bot-ear rp-ear-right"></div>
            <div class="rp-bot-shell">
              <div class="rp-bot-screen">
                <i class="rp-eye rp-eye-left"></i>
                <i class="rp-eye rp-eye-right"></i>
                <i class="rp-smile"></i>
              </div>
              <div class="rp-bot-mark">RP</div>
            </div>
            <div class="rp-bot-ring rp-ring-one"></div>
            <div class="rp-bot-ring rp-ring-two"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
