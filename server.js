const path = require('path');
const express = require('express');
const app = express();
app.get('/*all', function(req, res) {
  res.sendFile(path.join(__dirname + '/dist/index.html'));
});