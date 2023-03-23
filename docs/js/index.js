var score = 0;
var attempts = 0;
var mosaic_max_elem = 24;
var currently_displayed = [];
var available_types = ['original', 'factual', 'counterfactual'];

right_path = 'pool/right.jpeg'
wrong_path = 'pool/wrong.jpeg'


var counterfactual_filenames = [
  "0X100009310A3BD7FC.gif",
  "0X125C10BA973D3B07.gif",
  "0X13ECF4B877219D8B.gif",
  "0X1564DE8D0D40F195.gif",
  "0X1729DA199F862403.gif",
  "0X10094BA0A028EAC3.gif",
  "0X1274EA614B0388C1.gif",
  "0X140AAFB14918E8A4.gif",
  "0X15672D90E3EC7303.gif",
  "0X172D986B9780F9DE.gif",
  "0X1013E8A4864781B.gif",
  "0X127D3AEEA73EDE76.gif",
  "0X142EFADA9AF9DC99.gif",
  "0X159BDA520C61736A.gif",
  "0X172F640B239CBFCA.gif",
  "0X1039B49145DF4F25.gif",
  "0X12807854DFA9CC01.gif",
  "0X1459E82A63876D49.gif",
  "0X15C85161F8F9A55B.gif",
  "0X17458D83D9F9B188.gif",
  "0X105039B849CDB1CD.gif",
  "0X129682A2ECCD415.gif",
  "0X1469FC68080F968F.gif",
  "0X15C904A855E4FF2B.gif",
  "0X175303C2D48B91CA.gif",
  "0X10AD385C206C85C.gif",
  "0X12B890B1E2E14CC4.gif",
  "0X146DFEE92CCEC4E2.gif",
  "0X15D9F9EC9BA764E7.gif",
  "0X1763FB53ABCFB176.gif",
  "0X10B063E16BA7BEE3.gif",
  "0X12D7CFBEBDF0EAAF.gif",
  "0X148B59B7E2DE8071.gif",
  "0X15DA8D60960ABB2B.gif",
  "0X17BC4EF4BF83368B.gif",
  "0X10E47D630E15DE12.gif",
  "0X12F535C804FC3F13.gif",
  "0X148BFBF596042FF5.gif",
  "0X15E8BE2AE8C05C88.gif",
  "0X17DA9A44189C6A38.gif",
  "0X1108EED6EC60971E.gif",
  "0X13078849A6B5C581.gif",
  "0X14A6F87817547A95.gif",
  "0X160C233C08E7E8DE.gif",
  "0X17F29C21E6144BAB.gif",
  "0X110A2E1C465B1B53.gif",
  "0X132E461F48559F6B.gif",
  "0X14AD611ACCAD1161.gif",
  "0X1655D3F044EB6BB2.gif",
  "0X17F6D551F5734D8B.gif",
  "0X11116A740B44446A.gif",
  "0X1337E8945A141439.gif",
  "0X14C5B3A3DFECAC76.gif",
  "0X165EC58045A8A8BF.gif",
  "0X180951979820CA64.gif",
  "0X1112442D58D48C79.gif",
  "0X133BD3C195F9EA2.gif",
  "0X14E98112C64DA88B.gif",
  "0X166B717BBC2ECADA.gif",
  "0X18277456188D3B13.gif",
  "0X113195610E41EF2.gif",
  "0X133F50BB99C6E09.gif",
  "0X14F0C15097590562.gif",
  "0X166BF85033848E4E.gif",
  "0X1840607E4AF92B3E.gif",
  "0X113E9F83E605C0F5.gif",
  "0X13B0A1DABEDA5EDF.gif",
  "0X14F4B815A5210B8A.gif",
  "0X1677A8B7AAA975A1.gif",
  "0X1855EB02B676DC2.gif",
  "0X1192B98E520E87C8.gif",
  "0X13B4430D1930FF80.gif",
  "0X14FC7EEB6ACFE0BF.gif",
  "0X16902A1B1A139923.gif",
  "0X185672E7C8FF1212.gif",
  "0X11A07A83B21CFE1E.gif",
  "0X13D9B0A4BC16701D.gif",
  "0X1502E38D5D99E2E4.gif",
  "0X16AF26F9A372EEDE.gif",
  "0X1876806A464A0093.gif",
  "0X11BDF610427B903F.gif",
  "0X13DEAEAFDF1338AD.gif",
  "0X15146FB55B5697A0.gif",
  "0X16B85222D7B28245.gif",
  "0X18B8CD7AD450F707.gif",
  "0X11C89001BEF939E2.gif",
  "0X13E043A35E3EB490.gif",
  "0X15213080EC9FA05.gif",
  "0X16FA4A372C724D22.gif",
  "0X18C711A3AC175647.gif",
  "0X11C9572ABEDC9715.gif",
  "0X13E260DF39629AC9.gif",
  "0X153A76426320A57B.gif",
  "0X170DDD15D26D5F83.gif",
  "0X18D2B28FBD7DD266.gif",
  "0X11F798A4F0181ACF.gif",
  "0X13E488CF5C7C934C.gif",
  "0X155E07DE20CF09F5.gif",
  "0X17197B4B5AE30CC7.gif",
  "0X19001827BA355232.gif",
];

var factual_filenames = [
  "0X100009310A3BD7.gif",
  "0X11C89001BEF939.gif",
  "0X13B4430D1930FF.gif",
  "0X14E98112C64DA8.gif",
  "0X160C233C08E7E8.gif",
  "0X1763FB53ABCFB1.gif",
  "0X10094BA0A028EA.gif",
  "0X11C9572ABEDC97.gif",
  "0X13D9B0A4BC1670.gif",
  "0X14F0C150975905.gif",
  "0X1655D3F044EB6B.gif",
  "0X17BC4EF4BF8336.gif",
  "0X1013E8A486478.gif",
  "0X11F798A4F0181A.gif",
  "0X13DEAEAFDF1338.gif",
  "0X14F4B815A5210B.gif",
  "0X165EC58045A8A8.gif",
  "0X17DA9A44189C6A.gif",
  "0X1039B49145DF4F.gif",
  "0X125C10BA973D3B.gif",
  "0X13E043A35E3EB4.gif",
  "0X14FC7EEB6ACFE0.gif",
  "0X166B717BBC2ECA.gif",
  "0X17F29C21E6144B.gif",
  "0X105039B849CDB1.gif",
  "0X1274EA614B0388.gif",
  "0X13E260DF39629A.gif",
  "0X1502E38D5D99E2.gif",
  "0X166BF85033848E.gif",
  "0X17F6D551F5734D.gif",
  "0X10AD385C206C8.gif",
  "0X127D3AEEA73EDE.gif",
  "0X13E488CF5C7C93.gif",
  "0X15146FB55B5697.gif",
  "0X1677A8B7AAA975.gif",
  "0X180951979820CA.gif",
  "0X10B063E16BA7BE.gif",
  "0X12807854DFA9CC.gif",
  "0X13ECF4B877219D.gif",
  "0X15213080EC9FA.gif",
  "0X16902A1B1A1399.gif",
  "0X18277456188D3B.gif",
  "0X10E47D630E15DE.gif",
  "0X129682A2ECCD4.gif",
  "0X140AAFB14918E8.gif",
  "0X153A76426320A5.gif",
  "0X16AF26F9A372EE.gif",
  "0X1840607E4AF92B.gif",
  "0X1108EED6EC6097.gif",
  "0X12B890B1E2E14C.gif",
  "0X142EFADA9AF9DC.gif",
  "0X155E07DE20CF09.gif",
  "0X16B85222D7B282.gif",
  "0X1855EB02B676D.gif",
  "0X110A2E1C465B1B.gif",
  "0X12D7CFBEBDF0EA.gif",
  "0X1459E82A63876D.gif",
  "0X1564DE8D0D40F1.gif",
  "0X16FA4A372C724D.gif",
  "0X185672E7C8FF12.gif",
  "0X11116A740B4444.gif",
  "0X12F535C804FC3F.gif",
  "0X1469FC68080F96.gif",
  "0X15672D90E3EC73.gif",
  "0X170DDD15D26D5F.gif",
  "0X1876806A464A00.gif",
  "0X1112442D58D48C.gif",
  "0X13078849A6B5C5.gif",
  "0X146DFEE92CCEC4.gif",
  "0X159BDA520C6173.gif",
  "0X17197B4B5AE30C.gif",
  "0X18B8CD7AD450F7.gif",
  "0X113195610E41E.gif",
  "0X132E461F48559F.gif",
  "0X148B59B7E2DE80.gif",
  "0X15C85161F8F9A5.gif",
  "0X1729DA199F8624.gif",
  "0X18C711A3AC1756.gif",
  "0X113E9F83E605C0.gif",
  "0X1337E8945A1414.gif",
  "0X148BFBF596042F.gif",
  "0X15C904A855E4FF.gif",
  "0X172D986B9780F9.gif",
  "0X18D2B28FBD7DD2.gif",
  "0X1192B98E520E87.gif",
  "0X133BD3C195F9E.gif",
  "0X14A6F87817547A.gif",
  "0X15D9F9EC9BA764.gif",
  "0X172F640B239CBF.gif",
  "0X19001827BA3552.gif",
  "0X11A07A83B21CFE.gif",
  "0X133F50BB99C6E.gif",
  "0X14AD611ACCAD11.gif",
  "0X15DA8D60960ABB.gif",
  "0X17458D83D9F9B1.gif",
  "0X11BDF610427B90.gif",
  "0X13B0A1DABEDA5E.gif",
  "0X14C5B3A3DFECAC.gif",
  "0X15E8BE2AE8C05C.gif",
  "0X175303C2D48B91.gif",
];

var original_files = factual_filenames;

function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

class OneUs {
  constructor(type, filename, position) {
    this.type = type;
    this.filename = filename;
    this.real = (type == 'original');
    this.path = 'pool/' + this.type + '/' + this.filename;
    this.position = position;
  }

  display_game() {
    // Need to add call to game function
    // <div>'+capitalizeFirstLetter(this.type)+'</div>\

    return '<div class="video_wrapper">\
              <div class="video_container">\
                <img src="'+this.path+'" width="224">\
                <div class="caption">\
                  <button onclick="check_good(true, '+this.position+')">✅</button>\
                  <button onclick="check_good(false, '+this.position+')">❌</button>\
                </div>\
              </div>\
            </div>';
  }

  display_static() {
    return '<div class="video_wrapper">\
              <div class="video_container">\
                <img src="'+this.path+'" width="224">\
                <div class="caption">\
                < div>'+capitalizeFirstLetter(this.type)+'</div>\
                </div>\
              </div>\
            </div>';
  }
}

function pick_random() {
  // randomly choose between original, factual and counterfactual
  const random = Math.floor(Math.random() * available_types.length);
  const type = available_types[random];

  switch (type) {
    case 'original':
      var picked_list = original_files;
      break;
    case 'factual':
      var picked_list = factual_filenames;
      break;
    case 'counterfactual':
      var picked_list = counterfactual_filenames;
      break;
  }

  // randomly choose a filename
  const random2 = Math.floor(Math.random() * picked_list.length);
  const filename = picked_list[random2];

  return [type, filename];
}

function check_good(pred, index) {
  // Check if the prediction is correct
  cliked_element = currently_displayed[index];
  attempts += 1;
  if (cliked_element.real == pred) {
    score += 1;
    image_path = right_path
  }else{
    image_path = wrong_path
  }

  // Update score
  document.getElementById('score').innerHTML = 'Score: ' + score + '/' + attempts + ' (' + Math.round(score/attempts*100) + '%)';

  // Generate new US object in JS
  const [type, filename] = pick_random();
  const new_one = new OneUs(type, filename, index);
  currently_displayed[index] = new_one;

  // Display new US in HTML
  // Remove caption
  document.getElementById('gif-mosaic').children[index].children[0].children[1].remove();
  // Change gif for image
  document.getElementById('gif-mosaic').children[index].children[0].children[0].src = image_path;

  // Replace the element after 1s
  var tmp_div = document.createElement('div');
  tmp_div.innerHTML = new_one.display_game();

  setTimeout(()=> {
    document.getElementById('gif-mosaic').replaceChild(
      tmp_div.children[0],
      document.getElementById('gif-mosaic').children[index]
    );
  }, 1500);

}

function start_game() {
  currently_displayed = [];
  document.getElementById('gif-mosaic').innerHTML = '';
  document.getElementById('start_button').remove();
  document.getElementById('score').style.display = 'inline-block';

  // initial selection of gifs
  for (var i = 0; i < mosaic_max_elem; i++) {

    // use the pick_random function
    const [type, filename] = pick_random();

    // create the object
    const oneus = new OneUs(type, filename, i);
    currently_displayed.push(oneus);

    // display the gif
    document.getElementById('gif-mosaic').innerHTML += oneus.display_game();
  }
}


window.onload = (event) => {
  console.log('Page is fully loaded');
  currently_displayed = [];
  document.getElementById('gif-mosaic').innerHTML = '';

  // initial selection of gifs
  for (var i = 0; i < mosaic_max_elem; i++) {

    // use the pick_random function
    const [type, filename] = pick_random();

    // create the object
    const oneus = new OneUs(type, filename, i);
    currently_displayed.push(oneus);

    // display the gif
    document.getElementById('gif-mosaic').innerHTML += oneus.display_static();
  }
};