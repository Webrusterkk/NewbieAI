const path = require('path');
app.get('/*all', function(req, res) {
  res.sendFile(path.join(__dirname + '/dist/index.html'));
});