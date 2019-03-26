/*
 * Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
 *
 * This work is licensed under the Creative Commons Attribution-NonCommercial
 * 4.0 International License. To view a copy of this license, visit
 * http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
 * Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
 */

function sendRequest() {
  if(debug) console.log('Request is sent.');
  $('#generation-time').text('  (generating...)')
  return new Promise((resolve, reject) => {
    if(requestId) {
      clearTimeout(requestId);
      requestId = null;
    }
    requestId = null;
    $.ajax({
      method: 'POST',
      url: GuaGANHost + '/generate',
      data: {
        style: $('#clipboard').get(0).toDataURL(),
        semantic: $('#sketchpad').get(0).toDataURL()
      }
    }).done((data) => {
      let promise = drawDataUrlOnCanvas(data.generated_img, 'screen');
      $('#generation-time').text('  (in ' + data.generation_time.toFixed(2) + ' sec)')
      if(debug) console.log(data);
      resolve(promise);
    }).fail((error) => {
      reject(error);
    });
  });
}
      
function requestRandomImages(sketchpad) {
  return new Promise((resolve, reject) => {
    $.ajax({
      url: GuaGANHost + '/random/cocostuff'
    }).done((data) => {
    let promise = Promise.all([
      drawDataUrlOnCanvas(data.image, 'clipboard'),
      drawDataUrlOnCanvas(data.annotation, 'sketchpad')
    ]);
    let colorList = [], labelList = [];
    data.color_list.forEach(([color_array, label_text]) => {
      colorList.push(toColor(color_array));
      labelList.push(label_text);
    });
    sketchpad.setPalette(colorList, labelList);
    if(debug) console.log(data);
      resolve(promise);
    }).fail((error) => {
      reject(error);
    });
  });
}

function SketchPad(sketchpadId, paletteId, callback) {
  [this.canvas, this.ctx] = getCanvas(sketchpadId);
  this.palette = $('#'+paletteId);
  this.ctx.scale(1, 1);
    
  this.lineWidth = 3;
  this.lineColor = '#FF0000';
  this.nRows = 8;
    
  this.canvas.addEventListener('mousemove', (e) => {
    const rect = e.target.getBoundingClientRect();
    mouse.x = Math.floor(e.pageX - rect.x);
    mouse.y = Math.floor(e.pageY - rect.y - window.scrollY);
  }, false);
    
  this.canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.lineJoin = 'round';
    this.ctx.lineCap = 'round';
    this.ctx.strokeStyle = this.lineColor;
    this.ctx.beginPath();
    this.ctx.moveTo(mouse.x, mouse.y);
    this.canvas.addEventListener('mousemove', onPaint, false);
    if(requestId) {
      clearTimeout(requestId);
      requestId = null;
    }
  }, false);
    
  window.addEventListener('mouseup', (e) => {
    if(isDrawing) {
      requestId = setTimeout(callback, 1000);
    }
    isDrawing = false;
    this.canvas.removeEventListener('mousemove', onPaint, false);
  }, false);
    
  const onPaint = () => {
    this.ctx.lineTo(mouse.x, mouse.y);
    this.ctx.stroke();
  }
    
  $('#sketchpad').dblclick(() => {
    if(requestId) {
      clearTimeout(requestId);
      requestId = null;
    }
    this.fillArea();
    sendRequest();
  });
}
  
// http://www.williammalone.com/articles/html5-canvas-javascript-paint-bucket-tool/
SketchPad.prototype.fillArea = function() {
  const canvasWidth = 256-1, canvasHeight = 256-1;
  let origmgData = this.ctx.getImageData(0, 0, canvasWidth, canvasHeight);
  let imgData = this.ctx.getImageData(0, 0, canvasWidth, canvasHeight);
  let pixelStack = [[mouse.x, mouse.y]];
  let startR = null, startG = null, startB = null,
      [fillColorR, fillColorG, fillColorB] = parseColor(this.lineColor);
  while(pixelStack.length) {
    let newPos, x, y, pixelPos, reachLeft, reachRight;
    newPos = pixelStack.pop();
    [x, y] = newPos;
    pixelPos = (y*canvasWidth + x) * 4;
    if(startR == null) {
      startR = imgData.data[pixelPos];
      startG = imgData.data[pixelPos+1];
      startB = imgData.data[pixelPos+2];
      if(startR==fillColorR && startG==fillColorG && startB==fillColorB)
        break;
    }
    while(y-- >= 0 && matchStartColor(pixelPos)) {
      pixelPos -= canvasWidth * 4;
    }
    pixelPos += canvasWidth * 4;
    ++y;
    reachLeft = false;
    reachRight = false;
    while(y++ < canvasHeight-1 && matchStartColor(pixelPos)) {
      colorPixel(pixelPos);
      if(x > 0) {
        if(matchStartColor(pixelPos - 4)) {
          if(!reachLeft){
            pixelStack.push([x - 1, y]);
            reachLeft = true;
          }
        } else if(reachLeft) {
          reachLeft = false;
        }
      }

       if(x < canvasWidth-1) {
        if(matchStartColor(pixelPos + 4)) {
          if(!reachRight) {
            pixelStack.push([x + 1, y]);
            reachRight = true;
          }
        } else if(reachRight) {
          reachRight = false;
        }
      }
      pixelPos += canvasWidth * 4;
    }
  }
  this.ctx.putImageData(imgData, 0, 0);

  function matchStartColor(pixelPos){
    let r = imgData.data[pixelPos];	
    let g = imgData.data[pixelPos+1];	
    let b = imgData.data[pixelPos+2];
    return (r == startR && g == startG && b == startB);
  }

   function colorPixel(pixelPos) {
    imgData.data[pixelPos] = fillColorR;
    imgData.data[pixelPos+1] = fillColorG;
    imgData.data[pixelPos+2] = fillColorB;
    imgData.data[pixelPos+3] = 255;
  }
}

SketchPad.prototype.setPalette = function(colorList, labelList) {
  $('.row').empty();
  let color=null, label=null, row=null;
  for(let i=0; i<colorList.length; i++) {
    color = colorList[i];
    label = labelList[i];
    row = Math.floor(i / this.nRows) + 1;
    let div = $('<div>').addClass('sample').append(
      $('<div>').addClass('color-block')
                .css('background-color', color)
                .on('click', (e) => {
                  this.setPenColor($(e.target).css('background-color'));
                }))
              .append(
      $('<p>').addClass('color-label')
              .text(label))
              .appendTo($('.row:nth-child(' + row + ')'));
    if(label == 'unlabeled') {
      div.addClass('disabled')
         .find('.color-block')
         .off('click');
    }
  }
  if(colorList.length % this.nRows > 0) {
    for(let i=0; i < this.nRows - colorList.length % this.nRows; i++) {
      $('<div>').addClass('hidden-sample')
                .appendTo($('.row:nth-child(' + row + ')'));
    }
  }
  this.setPenColor(colorList[0]);
}

SketchPad.prototype.setPenColor = function(color) {
  $('.sample').each((index, element) => {
    if($(element).find('div').css('background-color') == color) {
      this.lineColor = color;
      $(element).addClass('active');
    } else {
      $(element).removeClass('active');
    }
  });
}

SketchPad.prototype.setPenWidth = function(width) {
  this.lineWidth = width;
}

function getCanvas(elementId) {
  const canvas = document.getElementById(elementId);
  const ctx = canvas.getContext('2d');
  return [canvas, ctx];
}
      
function toColor(values) {
  return 'rgb(' + values.join(', ') + ')';
}
    
function parseColor(rgb) {
  return rgb.replace(/[^\d,]/g, '').split(',').map(x => parseInt(x));
}

function cropResizeDataURL(datas, desiredSize) {
  return new Promise((resolve, reject) => {
    var img = document.createElement('img');
    img.onload = () => {
      const w = img.width, h = img.height;
      const ratio = 256 / Math.min(w, h);
      const w_draw = w * ratio, h_draw = h * ratio,
            x_draw = (256 - w_draw) / 2, y_draw = (256 - h_draw) / 2;
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = desiredSize;
      canvas.height = desiredSize;
      ctx.drawImage(img, x_draw, y_draw, w_draw, h_draw);
      const dataURI = canvas.toDataURL();
      resolve(dataURI);
    };
  img.src = datas;
  });
}

function drawDataUrlOnCanvas(dataUrl, canvasId) {
  return new Promise((resolve, reject) => {
    const [canvas, ctx] = getCanvas(canvasId);
    const img = new Image;
    img.src = dataUrl;
    img.addEventListener('load', (event) => {
      ctx.drawImage(img, 0, 0);
      resolve(canvasId)
    });
  });
}

function getUrlVars() {
  let vars = {};
  let parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi,    
    function(m, key, value) {
      vars[key] = value;
    });
  return vars;
}