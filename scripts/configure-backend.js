const fs = require('fs');
const path = require('path');

const mainRequirementsPath = path.join(__dirname, '..', 'backend', 'requirements.txt');
const cpuRequirementsPath = path.join(__dirname, '..', 'backend', 'requirements-cpu.txt');

// This is a simplified version of the script.
// A more complete version would use a library like `inquirer` to prompt the user.
// For now, we'll default to the CPU requirements.

fs.readFile(cpuRequirementsPath, 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading CPU requirements file:', err);
    return;
  }

  fs.appendFile(mainRequirementsPath, `
${data}`, (err) => {
    if (err) {
      console.error('Error appending to main requirements file:', err);
      return;
    }
    console.log('Successfully configured backend for CPU.');
  });
});
