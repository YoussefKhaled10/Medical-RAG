# RecoveryPath AI Approved Auth Design

Replace:

- `frontend/components/auth.py`
- `frontend/styles/auth.css`

Included:

- English-only Login, Create Account, and OTP Verification pages
- Approved navy, teal, blue, and white RecoveryPath visual style
- No Google sign-in
- No terms-and-conditions checkbox
- No forgot-password link
- Existing APIClient login, send-otp, register, profile, and logout integration

Run:

```bash
python -m py_compile frontend/components/auth.py
streamlit run frontend/app.py
```
