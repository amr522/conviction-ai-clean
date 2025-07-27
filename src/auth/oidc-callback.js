const express = require('express');
const axios = require('axios');
const jwt = require('jsonwebtoken');
const router = express.Router();

// OIDC callback handler
router.get('/auth/callback', async (req, res) => {
  try {
    const { code, state } = req.query;
    
    if (!code) {
      return res.status(400).json({ error: 'Authorization code missing' });
    }

    // Exchange authorization code for tokens
    const tokenResponse = await axios.post(
      `${process.env.OIDC_ISSUER_URL}/protocol/openid-connect/token`,
      new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: process.env.OIDC_CLIENT_ID,
        client_secret: process.env.OIDC_CLIENT_SECRET,
        code: code,
        redirect_uri: process.env.OIDC_REDIRECT_URL || `${req.protocol}://${req.get('host')}/auth/callback`
      }),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );

    const { access_token, id_token } = tokenResponse.data;

    // Decode and validate ID token
    const decoded = jwt.decode(id_token);
    
    if (!decoded) {
      return res.status(401).json({ error: 'Invalid ID token' });
    }

    // Set secure session cookie
    res.cookie('auth_token', access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 24 * 60 * 60 * 1000
    });

    // Set user info cookie
    res.cookie('user_info', JSON.stringify({
      sub: decoded.sub,
      email: decoded.email,
      name: decoded.name || decoded.preferred_username,
      groups: decoded.groups || []
    }), {
      httpOnly: false,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 24 * 60 * 60 * 1000
    });

    const redirectTo = state ? decodeURIComponent(state) : '/';
    res.redirect(redirectTo);

  } catch (error) {
    console.error('OIDC callback error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Authentication failed' });
  }
});

// Logout handler
router.post('/auth/logout', (req, res) => {
  res.clearCookie('auth_token');
  res.clearCookie('user_info');
  
  const logoutUrl = `${process.env.OIDC_ISSUER_URL}/protocol/openid-connect/logout?redirect_uri=${encodeURIComponent(`${req.protocol}://${req.get('host')}/`)}`;
  res.json({ logoutUrl });
});

// User info endpoint
router.get('/auth/user', (req, res) => {
  const userInfo = req.cookies.user_info;
  
  if (!userInfo) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  
  try {
    res.json(JSON.parse(userInfo));
  } catch (error) {
    res.status(500).json({ error: 'Invalid user info' });
  }
});

module.exports = router;