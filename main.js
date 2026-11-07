const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aTexCoord;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vTexCoord;

void main() {
  vec4 p = uModelView * vec4(aPosition, 1.0);
  vPosition = p.xyz;
  vNormal = normalize(uNormalMatrix * aNormal);
  vTangent = mat3(uModelView) * aTangent;
  vTexCoord = aTexCoord;
  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vTexCoord;

uniform sampler2D uTexture;
uniform sampler2D uNormalMap;
uniform sampler2D uHeightMap;
uniform int uUseTexture;
uniform int uUseNormalMap;
uniform int uUseBumpMap;
uniform float uBumpScale;
uniform vec3 uLightPos1;   // eye space1
uniform vec3 uLightPos2;   // eye space2
uniform vec3 uLightColor1;
uniform vec3 uLightColor2;
uniform vec3 uColor;
uniform float uDiffuseStrength;
uniform float uSpecularStrength;
uniform int uShadingMode; // 0: smooth, 1: flat

out vec4 fragColor;

vec3 illuminate(vec3 lightPos, vec3 lightColor, vec3 N, vec3 V) {
  vec3 L = normalize(lightPos - vPosition);
  float diff = max(dot(N, L), 0.0);
  vec3 R = reflect(-L, N);
  float spec = 0.0;
  if (diff > 0.0) {
    spec = pow(max(dot(V, R), 0.0), 32.0);
  }
  vec3 diffuseTerm = uColor * uDiffuseStrength * diff;
  vec3 specularTerm = vec3(uSpecularStrength * spec);
  return lightColor * (diffuseTerm + specularTerm);
}

void main() {
  vec3 smoothNormal = normalize(vNormal);
  vec3 N = smoothNormal;

  if (uShadingMode == 1) {
    vec3 faceNormal = normalize(cross(dFdx(vPosition), dFdy(vPosition)));
    if (dot(faceNormal, smoothNormal) < 0.0) faceNormal = -faceNormal;
    N = faceNormal;
  }

  // Build the tangent frame in eye space. Re-orthogonalize T against N
  // to account for model transformations and interpolation.
  vec3 T = normalize(vTangent - dot(vTangent, N) * N);
  vec3 B = normalize(cross(N, T));
  mat3 TBN = mat3(T, B, N);

  // Normal mapping: RGB values encode a tangent-space normal in [0,1].
  if (uUseNormalMap == 1) {
    vec3 mappedNormal = texture(uNormalMap, vTexCoord).rgb * 2.0 - 1.0;
    N = normalize(TBN * mappedNormal);
  }

  // Bump mapping: use finite differences of a grayscale height map
  // to perturb the normal without changing the mesh geometry.
  if (uUseBumpMap == 1) {
    vec2 texel = 1.0 / vec2(textureSize(uHeightMap, 0));
    float hL = dot(texture(uHeightMap, vTexCoord - vec2(texel.x, 0.0)).rgb,
                   vec3(0.299, 0.587, 0.114));
    float hR = dot(texture(uHeightMap, vTexCoord + vec2(texel.x, 0.0)).rgb,
                   vec3(0.299, 0.587, 0.114));
    float hD = dot(texture(uHeightMap, vTexCoord - vec2(0.0, texel.y)).rgb,
                   vec3(0.299, 0.587, 0.114));
    float hU = dot(texture(uHeightMap, vTexCoord + vec2(0.0, texel.y)).rgb,
                   vec3(0.299, 0.587, 0.114));
    vec3 bumpNormal = normalize(vec3(-(hR - hL) * uBumpScale,
                                     -(hU - hD) * uBumpScale, 1.0));
    N = normalize(TBN * bumpNormal);
  }

  vec3 V = normalize(-vPosition);
  vec3 surfaceColor = uColor;
  if (uUseTexture == 1) surfaceColor *= texture(uTexture, vTexCoord).rgb;

  vec3 ambient = 0.12 * surfaceColor;
  vec3 L1 = normalize(uLightPos1 - vPosition);
  vec3 L2 = normalize(uLightPos2 - vPosition);
  float d1 = max(dot(N, L1), 0.0);
  float d2 = max(dot(N, L2), 0.0);
  vec3 R1 = reflect(-L1, N);
  vec3 R2 = reflect(-L2, N);
  float s1 = d1 > 0.0 ? pow(max(dot(V, R1), 0.0), 32.0) : 0.0;
  float s2 = d2 > 0.0 ? pow(max(dot(V, R2), 0.0), 32.0) : 0.0;
  vec3 diffuse = surfaceColor * uDiffuseStrength *
      (uLightColor1 * d1 + uLightColor2 * d2);
  vec3 specular = uSpecularStrength *
      (uLightColor1 * s1 + uLightColor2 * s2);
  fragColor = vec4(ambient + diffuse + specular, 1.0);
}
`;

// WebGL setup
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl2");

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

// Fixed attribute slots, so the meshes stay valid when the shaders are recompiled
const ATTRIBUTES = { aPosition: 0, aNormal: 1, aTexCoord: 2, aTangent: 3 };

function createProgram(vsSource, fsSource) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
  for (const name in ATTRIBUTES) gl.bindAttribLocation(prog, ATTRIBUTES[name], name);
  gl.linkProgram(prog);
  // gl.deleteShader(compile(gl.VERTEX_SHADER, vsSource));
  // gl.deleteShader(compile(gl.FRAGMENT_SHADER, fsSource));
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Shader editors
const vertEditor = document.getElementById("vertEditor");
const fragEditor = document.getElementById("fragEditor");
const shaderError = document.getElementById("shaderError");
vertEditor.value = VERTEX_SHADER;
fragEditor.value = FRAGMENT_SHADER;

let program;
let uniforms = {};
function updateProgram() {
  try {
    const nextProgram = createProgram(vertEditor.value, fragEditor.value);
    if (program) gl.deleteProgram(program);
    program = nextProgram;
    gl.useProgram(program);
    const names = [
      "uModelView", "uProjection", "uNormalMatrix", "uLightPos1", "uLightPos2",
      "uLightColor1", "uLightColor2", "uColor", "uDiffuseStrength",
      "uSpecularStrength", "uShadingMode", "uTexture", "uUseTexture",
      "uNormalMap", "uHeightMap", "uUseNormalMap", "uUseBumpMap", "uBumpScale"
    ];
    uniforms = Object.fromEntries(names.map(name => [name, gl.getUniformLocation(program, name)]));
    gl.uniform1i(uniforms.uTexture, 0);
    shaderError.textContent = "";
  } catch (e) {
    // Keep the last working program active if a live shader edit has an error.
    shaderError.textContent = e.message.replace(/\0/g, "");
  }
}
updateProgram();
vertEditor.addEventListener("input", updateProgram);
fragEditor.addEventListener("input", updateProgram);

function bindAttribute(name, data, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.vertexAttribPointer(ATTRIBUTES[name], size, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(ATTRIBUTES[name]);
}

function createMesh({ positions, normals, tangents, texCoords, indices }) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  bindAttribute("aPosition", positions, 3);
  bindAttribute("aNormal", normals, 3);
  bindAttribute("aTangent", tangents, 3);
  bindAttribute("aTexCoord", texCoords, 2);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vao, count: indices.length };
}

const cubeMesh = createMesh(createCube());
const cylinderMesh = createMesh(createCylinder(32));

// Additional curved primitives for the texture-mapping assignment.
// They are uploaded now and will be added to the scene when their textures
// and object placements are introduced in the next stages.
const sphereMesh = createMesh(createSphere(24, 32));
const coneMesh = createMesh(createCone(32));
const torusMesh = createMesh(createTorus(0.72, 0.28, 48, 16));

// Load an image as a WebGL texture. A white placeholder is used until the
// image finishes loading, so the scene can render immediately.
function loadTexture(url) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                  gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
  };
  image.onerror = () => console.error(`Could not load texture: ${url}`);
  image.src = url;
  return texture;
}

const brickTexture = loadTexture("brick.png");
// Stage 4: color textures for the three curved primitives.
const helmetTexture = loadTexture("helmet.png");
const coneTexture = loadTexture("cone.png");
const tireTexture = loadTexture("tire.png");
// The professor-provided texture2.jpg is a tangent-space normal map.
const tireNormalMap = loadTexture("texture2.jpg");
// Reuse the brick image as a grayscale-derived height source for bump mapping.
const brickHeightMap = brickTexture;

// Stage 6: use an HTML video as a live WebGL texture.
// The muted video can autoplay and loops through the pit-stop light sequence.
const pitStopVideo = document.createElement("video");
pitStopVideo.src = "pit_stop_signal.mp4";
pitStopVideo.loop = true;
pitStopVideo.muted = true;
pitStopVideo.autoplay = true;
pitStopVideo.playsInline = true;
pitStopVideo.preload = "auto";
const pitStopVideoTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, pitStopVideoTexture);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
              gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([20, 25, 35, 255]));
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
pitStopVideo.addEventListener("canplay", () => {
  pitStopVideo.play().catch(error => console.warn("Pit-stop video autoplay was blocked:", error));
});
pitStopVideo.load();

function updateVideoTexture() {
  if (pitStopVideo.readyState < pitStopVideo.HAVE_CURRENT_DATA) return;
  gl.bindTexture(gl.TEXTURE_2D, pitStopVideoTexture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                gl.UNSIGNED_BYTE, pitStopVideo);
}


gl.enable(gl.DEPTH_TEST);
gl.clearColor(0.035, 0.035, 0.045, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld1 = [3, 4, 5];
const lightWorld2 = [-6, 2, 4];
const lightColor1 = [1.0, 0.92, 0.82];
const lightColor2 = [0.38, 0.55, 1.0];

// Interactive controls
let dragRotX = 0, dragRotY = -0.5;      // mouse drag
let camX = 0, camY = 0, camZ = -7;      // arrow keys, w, s
let sliderAngle = 0;                    // degrees
let spinning = false, spinAngle = 0;    // Start/Stop button

// New controls: a suspension joint, wheel rotation, and per-object materials.
let armAngleDegrees = 0;
let wheelAngleDegrees = 0;
let armDiffuseStrength = 0.8;
let wheelSpecularStrength = 0.85;
let shadingMode = 0;

// Mapping demonstration controls (used only by the tire and bracket).
let normalMappingEnabled = true;
let bumpMappingEnabled = true;
let bumpStrength = 8.0;

let mouseDown = false, lastX, lastY;
canvas.addEventListener("mousedown", e => { mouseDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", e => {
  if (!mouseDown) return;
  dragRotY += (e.clientX - lastX) * 0.01;
  dragRotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener("keydown", e => {
  if (e.target === vertEditor || e.target === fragEditor) return;  // let the editors use the keys
  const step = 0.2;
  switch (e.key) {
    case "ArrowUp": camY -= step; break;
    case "ArrowDown": camY += step; break;
    case "ArrowLeft": camX += step; break;
    case "ArrowRight": camX -= step; break;
    case "w": camZ += step; break;
    case "s": camZ -= step; break;
    default: return;
  }
  e.preventDefault();  // keep the arrow keys from scrolling the page
});

const axisSelect = document.getElementById("axisSelect");
document.getElementById("slider").addEventListener("input", e => {
  sliderAngle = Number(e.target.value);
  document.getElementById("sliderValue").textContent = sliderAngle;
});
document.getElementById("spinButton").addEventListener("click", () => spinning = !spinning);

document.getElementById("shadingSelect").addEventListener("change", e => {
  shadingMode = e.target.value === "flat" ? 1 : 0;
});
document.getElementById("armAngleSlider").addEventListener("input", e => {
  armAngleDegrees = Number(e.target.value);
  document.getElementById("armAngleValue").textContent = `${armAngleDegrees}°`;
});
document.getElementById("wheelAngleSlider").addEventListener("input", e => {
  wheelAngleDegrees = Number(e.target.value);
  document.getElementById("wheelAngleValue").textContent = `${wheelAngleDegrees}°`;
});
document.getElementById("armDiffuseSlider").addEventListener("input", e => {
  armDiffuseStrength = Number(e.target.value);
  document.getElementById("armDiffuseValue").textContent = armDiffuseStrength.toFixed(1);
});
document.getElementById("wheelSpecularSlider").addEventListener("input", e => {
  wheelSpecularStrength = Number(e.target.value);
  document.getElementById("wheelSpecularValue").textContent = wheelSpecularStrength.toFixed(1);
});

// Buttons for demonstrating normal mapping and bump mapping during grading.
function updateMappingControlLabels() {
  const normalButton = document.getElementById("normalMapToggle");
  const bumpButton = document.getElementById("bumpMapToggle");
  normalButton.textContent = `Tire Normal Mapping: ${normalMappingEnabled ? "ON" : "OFF"}`;
  bumpButton.textContent = `Bracket Bump Mapping: ${bumpMappingEnabled ? "ON" : "OFF"}`;
  normalButton.setAttribute("aria-pressed", String(normalMappingEnabled));
  bumpButton.setAttribute("aria-pressed", String(bumpMappingEnabled));
  document.getElementById("bumpStrengthValue").textContent = bumpStrength.toFixed(1);
}

// Use explicit button state assignments so every click has a predictable result.
// The status text also makes the current setting obvious during grading.
const normalMapButton = document.getElementById("normalMapToggle");
const bumpMapButton = document.getElementById("bumpMapToggle");
const bumpStrengthInput = document.getElementById("bumpStrengthSlider");
const mappingStatus = document.getElementById("mappingStatus");

function updateMappingControlLabels() {
  normalMapButton.textContent = `Tire Normal Mapping: ${normalMappingEnabled ? "ON" : "OFF"}`;
  bumpMapButton.textContent = `Bracket Bump Mapping: ${bumpMappingEnabled ? "ON" : "OFF"}`;
  normalMapButton.setAttribute("aria-pressed", String(normalMappingEnabled));
  bumpMapButton.setAttribute("aria-pressed", String(bumpMappingEnabled));
  document.getElementById("bumpStrengthValue").textContent = bumpStrength.toFixed(1);
  mappingStatus.textContent = `Normal map: ${normalMappingEnabled ? "ENABLED" : "DISABLED"} | Bump map: ${bumpMappingEnabled ? "ENABLED" : "DISABLED"} | Bump strength: ${bumpStrength.toFixed(1)}`;
}

normalMapButton.onclick = () => {
  normalMappingEnabled = !normalMappingEnabled;
  updateMappingControlLabels();
};
bumpMapButton.onclick = () => {
  bumpMappingEnabled = !bumpMappingEnabled;
  updateMappingControlLabels();
};
bumpStrengthInput.oninput = () => {
  bumpStrength = Number(bumpStrengthInput.value);
  updateMappingControlLabels();
};
document.getElementById("resetMappingButton").onclick = () => {
  normalMappingEnabled = true;
  bumpMappingEnabled = true;
  bumpStrength = 8.0;
  bumpStrengthInput.value = "8";
  updateMappingControlLabels();
};
updateMappingControlLabels();


function createNode(mesh, color, shape, material = {}) {
  return {
    mesh, color, shape, joint: null, children: [],
    texture: material.texture ?? null,
    normalMap: material.normalMap ?? null,
    heightMap: material.heightMap ?? null,
    bumpScale: material.bumpScale ?? 5.0,
    diffuseStrength: material.diffuseStrength ?? 0.8,
    specularStrength: material.specularStrength ?? 0.35,
    shadingMode: material.shadingMode ?? 0
  };
}

// // Lower arm: scale the cylinder then move it up by 0.6 so its bottom sits at the joint
// const lowerArm = createNode(cylinderMesh, [1, 0.2, 0.2],
//   multiply(translate(0, 0.6, 0), scale(0.3, 0.6, 0.3)));

// // Upper arm: scale the cube then move it up by 0.5 so its bottom sits at the joint
// const upperArm = createNode(cubeMesh, [0.2, 0.4, 1],
//   multiply(translate(0, 0.5, 0), scale(0.2, 0.5, 0.2)));

// F1 wheel and suspension assembly. Each child joint is expressed relative
// to its parent's joint, so rotations propagate down the hierarchy.
// Level 1: mounting bracket (root)
const bracket = createNode(cubeMesh, [1.0, 1.0, 1.0], scale(0.48, 0.22, 0.24), {
  diffuseStrength: 0.75, specularStrength: 0.15, shadingMode: 1,
  texture: brickTexture, heightMap: brickHeightMap, bumpScale: 8.0
});

// Level 2: upper suspension arm.
const upperArm = createNode(cylinderMesh, [0.72, 0.74, 0.78],
  multiply(translate(0, -0.48, 0), scale(0.11, 0.48, 0.11)),
  { diffuseStrength: 0.8, specularStrength: 0.3, shadingMode: 0 });

// Level 3: lower suspension arm.
const lowerArm = createNode(cylinderMesh, [0.82, 0.25, 0.12],
  multiply(translate(0, -0.38, 0), scale(0.085, 0.38, 0.085)),
  { diffuseStrength: 0.8, specularStrength: 0.25, shadingMode: 0 });

// Level 4: wheel hub
const wheelHub = createNode(cylinderMesh, [0.72, 0.75, 0.78],
  multiply(translate(0, 0, -0.055), multiply(rotateX(Math.PI / 2), scale(0.25, 0.075, 0.25))),
  { diffuseStrength: 0.75, specularStrength: 0.7, shadingMode: 0 });

// Level 5: tire
const wheel = createNode(cylinderMesh, [0.12, 0.13, 0.15],
  scale(0.52, 0.16, 0.52),
  { diffuseStrength: 0.8, specularStrength: 0.85, shadingMode: 0,
    texture: tireTexture, normalMap: tireNormalMap });

// Two spokes make wheel rotation visible. They are children of the tire.
const spokeA = createNode(cubeMesh, [0.78, 0.79, 0.82],
  multiply(translate(0, 0, 0.29), scale(0.34, 0.045, 0.025)),
  { diffuseStrength: 0.8, specularStrength: 0.6, shadingMode: 1 });
const spokeB = createNode(cubeMesh, [0.78, 0.79, 0.82],
  multiply(translate(0, 0, 0.29), scale(0.045, 0.34, 0.025)),
  { diffuseStrength: 0.8, specularStrength: 0.6, shadingMode: 1 });

// Build the hierarchy
bracket.children.push(upperArm);
upperArm.children.push(lowerArm);
lowerArm.children.push(wheelHub);

wheelHub.children.push(wheel, spokeA, spokeB);

// Stage 4: three independently textured curved objects placed around the
// wheel assembly. Their UVs come from createSphere(), createCone(), and
// createTorus() in models.js.
const helmetObject = createNode(
  sphereMesh, [1, 1, 1],
  multiply(translate(-1.65, 1.15, 0.15), scale(0.48, 0.55, 0.48)),
  { texture: helmetTexture, diffuseStrength: 0.9, specularStrength: 0.45 }
);

const coneObject = createNode(
  coneMesh, [1, 1, 1],
  multiply(translate(1.55, -0.72, 0.15), scale(0.38, 0.55, 0.38)),
  { texture: coneTexture, diffuseStrength: 0.95, specularStrength: 0.18 }
);

// The torus is rotated so its ring faces the camera, like a tire viewed
// from the side. The torus mesh itself is centered at the origin.
const tireObject = createNode(
  torusMesh, [1, 1, 1],
  multiply(translate(1.55, 0.78, 0.15),
    multiply(rotateX(Math.PI / 2), scale(0.48, 0.48, 0.48))),
  { texture: tireTexture, normalMap: tireNormalMap, diffuseStrength: 0.85, specularStrength: 0.28 }
);

// Animated video-textured pit-stop signal panel.
// The front (+Z) face of the cube displays the looping video.
const pitStopSignal = createNode(
  cubeMesh, [1, 1, 1],
  multiply(translate(-1.55, -0.78, 0.18), scale(0.68, 0.38, 0.055)),
  { texture: pitStopVideoTexture, diffuseStrength: 1.0, specularStrength: 0.0, shadingMode: 1 }
);

// Inverse-transpose of the upper-left 3x3 of a column-major mat4.
function makeNormalMatrix(m) {
  const a = m[0], b = m[4], c = m[8];
  const d = m[1], e = m[5], f = m[9];
  const g = m[2], h = m[6], i = m[10];
  const c00 = e * i - f * h;
  const c01 = f * g - d * i;
  const c02 = d * h - e * g;
  const c10 = c * h - b * i;
  const c11 = a * i - c * g;
  const c12 = b * g - a * h;
  const c20 = b * f - c * e;
  const c21 = c * d - a * f;
  const c22 = a * e - b * d;
  const det = a * c00 + b * c01 + c * c02;
  if (Math.abs(det) < 1e-8) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
  const invDet = 1 / det;
  // Store column-major for uniformMatrix3fv.
  return new Float32Array([
    c00 * invDet, c10 * invDet, c20 * invDet,
    c01 * invDet, c11 * invDet, c21 * invDet,
    c02 * invDet, c12 * invDet, c22 * invDet
  ]);
}

function drawNode(node, frame) {
  const modelView = multiply(frame, node.shape);
  gl.uniformMatrix4fv(uniforms.uModelView, false, modelView);
  gl.uniformMatrix3fv(uniforms.uNormalMatrix, false, makeNormalMatrix(modelView));
  gl.uniform3fv(uniforms.uColor, node.color);
  gl.uniform1f(uniforms.uDiffuseStrength,
    node === lowerArm ? armDiffuseStrength : node.diffuseStrength);
  gl.uniform1f(uniforms.uSpecularStrength,
    node === wheel ? wheelSpecularStrength : node.specularStrength);
  gl.uniform1i(uniforms.uShadingMode, shadingMode);
  gl.uniform1i(uniforms.uUseTexture, node.texture ? 1 : 0);
  gl.uniform1i(uniforms.uUseNormalMap, node.normalMap && normalMappingEnabled ? 1 : 0);
  gl.uniform1i(uniforms.uUseBumpMap, node.heightMap && bumpMappingEnabled ? 1 : 0);
  gl.uniform1f(uniforms.uBumpScale, bumpStrength);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, node.texture || null);
  gl.uniform1i(uniforms.uTexture, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, node.normalMap || null);
  gl.uniform1i(uniforms.uNormalMap, 1);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, node.heightMap || null);
  gl.uniform1i(uniforms.uHeightMap, 2);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindVertexArray(node.mesh.vao);
  gl.drawElements(gl.TRIANGLES, node.mesh.count, gl.UNSIGNED_SHORT, 0);
}

//Data structure
function drawTree(root, rootFrame) {
  const stack = [[root, rootFrame]];
  while (stack.length > 0) {
    const [node, parentFrame] = stack.pop();
    const frame = multiply(parentFrame, node.joint);
    drawNode(node, frame);
    for (const child of node.children) stack.push([child, frame]);
  }
}

// Render loop
let lastTime = 0;
function render(time) {
  const t = time / 1000;
  const delta = lastTime === 0 ? 0 : t - lastTime;
  if (spinning) spinAngle += delta;
  lastTime = t;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  updateVideoTexture();

  // Camera moves with the keys; the light stays fixed in the world.
  const view = translate(camX, camY, camZ);
  gl.uniformMatrix4fv(uniforms.uProjection, false, projection);
  gl.uniform3fv(uniforms.uLightPos1, transformPoint(view, lightWorld1));
  gl.uniform3fv(uniforms.uLightPos2, transformPoint(view, lightWorld2));
  gl.uniform3fv(uniforms.uLightColor1, lightColor1);
  gl.uniform3fv(uniforms.uLightColor2, lightColor2);

  // Whole-model rotation
  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);

  // Place the assembly near the center of the canvas.
  bracket.joint = multiply(translate(0, 0.9, 0), rotation);
  upperArm.joint = translate(0, -0.22, 0);
  lowerArm.joint = multiply(translate(0, -0.96, 0), rotateZ((-0.65 + armAngleDegrees * Math.PI / 180)));
  wheelHub.joint = translate(0, -0.76, 0);
  const wheelSpin = wheelAngleDegrees * Math.PI / 180 + spinAngle * 3;
  wheel.joint = multiply(translate(0, 0, 0.13),
    multiply(rotateX(Math.PI / 2), rotateY(wheelSpin)));
  spokeA.joint = rotateZ(wheelSpin);
  spokeB.joint = rotateZ(wheelSpin);

  drawTree(bracket, view);

  // Draw the additional textured objects in the same world/view frame.
  drawNode(helmetObject, view);
  drawNode(coneObject, view);
  drawNode(tireObject, view);
  drawNode(pitStopSignal, view);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
