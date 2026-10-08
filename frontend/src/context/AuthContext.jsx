import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { getCurrentUser, loginUser, registerUser } from "../api/authApi";
import { apiClient } from "../api/client";
import { AUTH_STORAGE_KEY } from "../constants";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => window.localStorage.getItem(AUTH_STORAGE_KEY) || "");
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(Boolean(token));

  const remember = useCallback((nextToken, nextUser) => {
    if (nextToken) window.localStorage.setItem(AUTH_STORAGE_KEY, nextToken);
    else window.localStorage.removeItem(AUTH_STORAGE_KEY);
    setToken(nextToken);
    setUser(nextUser);
  }, []);

  const signOut = useCallback(() => remember("", null), [remember]);

  // An expired or revoked token signs the user out.
  useEffect(() => {
    const id = apiClient.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error?.response?.status === 401 && window.localStorage.getItem(AUTH_STORAGE_KEY)) {
          signOut();
        }
        return Promise.reject(error);
      }
    );
    return () => apiClient.interceptors.response.eject(id);
  }, [signOut]);

  useEffect(() => {
    let active = true;
    // Without a token there is nothing to load (loading started as false).
    if (!token) return undefined;
    getCurrentUser()
      .then((me) => active && setUser(me))
      .catch(() => active && signOut())
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [token, signOut]);

  const value = useMemo(
    () => ({
      user,
      loading,
      signIn: async (credentials) => {
        const response = await loginUser(credentials);
        remember(response.access_token, response.user);
      },
      signUp: async (details) => {
        const response = await registerUser(details);
        remember(response.access_token, response.user);
      },
      signOut,
    }),
    [user, loading, remember, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
}
