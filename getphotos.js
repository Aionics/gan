const req = require('request');
const fs = require('fs')
const dataset = require('./imagedataset');

console.log(dataset.length);
dataset.forEach((link, i) => {
    req(link, { encoding: 'binary' }, function (error, response, body) {
        fs.writeFile('keyboards/' + i + '.jpg', body, 'binary', function (err) {
            console.log(err);
        });
    });
})
console.log('done');
// 247615810
// req("https://api.vk.com/method/photos.get?owner_id=-46860100&rev=1&album_id=247615810&photo_sizes=1&count=4000", (error, response, body) => {
//   let list = JSON.parse(body).response
//   let links = []
//
//   for (photoObj of list) {
//     let sizes = photoObj.sizes
//     for (size of sizes) {
//       if (size.width == 320) {
//         links.push(size.src)
//       }
//     }
//   }
//
//   links.forEach((link, i) => {
//     req(link, {encoding: 'binary'}, function(error, response, body) {
//       fs.writeFile('pixels2/' + (2000 + i) + '.jpg', body, 'binary', function (err) {});
//     });
//   })
// })
