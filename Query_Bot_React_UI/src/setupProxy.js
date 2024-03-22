const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/context_query',
    createProxyMiddleware({
      target: 'http://localhost:5001/context_query',
      changeOrigin: true,
    })
  );
  app.use(
    '/bert_query',
    createProxyMiddleware({
      target: 'http://localhost:5002/bert_query',
      changeOrigin: true,
    })
  );
};
