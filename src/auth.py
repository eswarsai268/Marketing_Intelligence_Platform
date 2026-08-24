import streamlit as st


def require_login():
    """Require Google authentication before showing the app."""

    # User is not logged in
    if not st.user.is_logged_in:
        st.title("🔐 Login Required")
        st.write("Please sign in with Google to continue.")

        if st.button("🔑 Login with Google"):
            st.login()

        st.stop()

    # User is logged in
    st.sidebar.success("✅ Logged in")

    # Safely display available user information
    user = st.user

    if hasattr(user, "name"):
        st.sidebar.write(f"Welcome, **{user.name}**")
    else:
        st.sidebar.write("Welcome!")

    if st.sidebar.button("Logout"):
        st.logout()