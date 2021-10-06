const fs = require('fs');
const util = require('util');
const path = require('path');

const readDir = util.promisify(fs.readdir);
const pStat = util.promisify(fs.stat);

const printJSFile = async dirName => {
    try {
        const files = await readDir(dirName);
        files.forEach( async file => {
            const filePath = path.join(dirName, file);
            const stat = await pStat(filePath);            
            
            if(stat.isDirectory()) printJSFile(filePath);                        
            else if(path.extname(file) == '.js') console.log(filePath);
        });       
    } catch (err) {
        console.error(err);
    }
};

printJSFile('test');