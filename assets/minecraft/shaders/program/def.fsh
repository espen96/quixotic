#version 150

uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec4 skycol;

uniform sampler2D DiffuseSampler;
in float skyIntensity;
in vec3 nsunColor;
in vec3 sunDir;
in float skyIntensityNight;
in float rainStrength;
in float sunIntensity;
in float moonIntensity;

in vec3 ambientUp;
in vec3 ambientLeft;
in vec3 ambientRight;
in vec3 ambientB;
in vec3 ambientF;
in vec3 ambientDown;
in vec3 lightSourceColor;
in vec3 sunColor;
in vec3 sunColorCloud;
in vec3 moonColor;
in vec3 moonColorCloud;
in vec3 zenithColor;
in vec3 avgSky;

in vec3 ds;
in vec3 ms;
out vec4 fragColor;
in vec4 lightCol;

#define PI 3.141592

#define MIN_LIGHT_AMOUNT 0.225
#define TORCH_AMOUNT 1.0
#define TORCH_R 1.0
#define TORCH_G 0.5
#define TORCH_B 0.2

#define SKY_BRIGHTNESS_DAY 0.4
#define SKY_BRIGHTNESS_NIGHT 2.0
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)

float facos(float inX) {

  const float C0 = 1.56467;
  const float C1 = -0.155972;

  float x = abs(inX);
  float res = C1 * x + C0;
  res *= sqrt(1.0f - x);

  return (inX >= 0) ? res : PI - res;
}

vec3 ScreenSpaceDither(vec2 vScreenPos) {
  vec3 vDither = vec3(dot(vec2(131.0, 312.0), vScreenPos.xy + fract(Time * 2048)));
  vDither.rgb = fract(vDither.rgb / vec3(103.0, 71.0, 97.0)) * vec3(2.0, 2.0, 2.0) - vec3(0.5, 0.5, 0.5);
  return (vDither.rgb / 15);
}

vec3 lumaBasedReinhardToneMapping(vec3 color) {
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float toneMappedLuma = luma / (1. + luma);
  color *= toneMappedLuma / luma;
  color = pow(color, vec3(1. / 2.2));
  return color;
}

vec3 reinhard_extended(vec3 v, float max_white) {
  vec3 numerator = v * (1.0f + (v / vec3(max_white * max_white)));
  return numerator / (1.0f + v);
}
vec3 reinhard(vec3 v) {
  return v / (1.0f + v);
}
float luminance(vec3 v) {
  return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
  float l_in = luminance(c_in);
  return c_in * (l_out / l_in);
}
vec3 reinhard_extended_luminance(vec3 v, float max_white_l) {
  float l_old = luminance(v);
  float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
  float l_new = numerator / (1.0f + l_old);
  return change_luminance(v, l_new);
}
vec3 reinhard_jodie(vec3 v) {
  float l = luminance(v);
  vec3 tv = v / (1.0f + v);
  return mix(v / (1.0f + l), tv, tv);
}

float luma3(vec3 color) {
  return dot(color, vec3(0.21, 0.72, 0.07));
}

void main() {
//vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
  vec3 avgAmbient = (ambientUp + ambientLeft + ambientRight + ambientB + ambientF + ambientDown) / 6. * (1.0 + rainStrength * 0.2);

  if(gl_FragCoord.x < 17. && gl_FragCoord.y < 17.) {

    vec3 avgAmbient = ds + ms;
    avgAmbient = mix(avgAmbient * vec3(0.2, 0.2, 0.5) * 2.0, avgAmbient, 1 - rainStrength);
    float lumC = luma3(avgAmbient.rgb);
    vec3 diff = avgAmbient.rgb - lumC;
    avgAmbient = avgAmbient.rgb + diff * (-lumC * 0.5 + 0.5);
    float skyLut = floor(gl_FragCoord.y) / 15.;
    float sky_lightmap = pow(skyLut, 4.5);
    float torchLut = floor(gl_FragCoord.x) / 15.;
    torchLut *= torchLut;

    float torch_lightmap = ((torchLut * torchLut) * (torchLut * torchLut)) * (torchLut * 10.0) + torchLut;
    float avgEyeIntensity = ((sunIntensity * 120.0 + moonIntensity * 4.0) + skyIntensity * 230.0 + skyIntensityNight * 3.0) * sky_lightmap;
    float exposure = 0.18 / log2(max(avgEyeIntensity * 0.16 + 1.0, 1.13));
    vec3 ambient = (((avgAmbient) * 20.0) * sky_lightmap * log2(1.13 + sky_lightmap * 1.5) + torch_lightmap * 0.05 * vec3(TORCH_R, TORCH_G, TORCH_B) * TORCH_AMOUNT) * exposure * vec3(1.0, 0.96, 0.96) + MIN_LIGHT_AMOUNT * 0.001 * vec3(0.75, 1.0, 1.25);

    fragColor = vec4(reinhard_jodie(ambient * 10.0), 1.0);

  }
//Save light values
  if(gl_FragCoord.x < 1. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientUp, 1.0);
  if(gl_FragCoord.x > 1. && gl_FragCoord.x < 2. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientDown, 1.0);
  if(gl_FragCoord.x > 2. && gl_FragCoord.x < 3. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientLeft, 1.0);
  if(gl_FragCoord.x > 3. && gl_FragCoord.x < 4. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientRight, 1.0);
  if(gl_FragCoord.x > 4. && gl_FragCoord.x < 5. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientB, 1.0);
  if(gl_FragCoord.x > 5. && gl_FragCoord.x < 6. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(ambientF, 1.0);
  if(gl_FragCoord.x > 6. && gl_FragCoord.x < 7. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(lightSourceColor, 1.0);
  if(gl_FragCoord.x > 7. && gl_FragCoord.x < 8. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(avgAmbient, 1.0);
  if(gl_FragCoord.x > 8. && gl_FragCoord.x < 9. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1) {
    float lumC = luma3(lightCol.rgb);
    vec3 diff = lightCol.rgb - lumC;
    vec3 lightCol = lightCol.rgb + diff * (-lumC * 0.2 + 0.7);
    fragColor.rgb = lightCol.rgb;
  }

  if(gl_FragCoord.x > 11. && gl_FragCoord.x < 12. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    fragColor = vec4(avgSky, 1.0);

  else if(gl_FragCoord.x > 18. && gl_FragCoord.y > 1.) {
    float cosY = clamp(floor(gl_FragCoord.x - 18.0) / 256. * 2.0 - 1.0, -1.0 + 1e-5, 1.0 - 1e-5);
    cosY = pow(abs(cosY), 1 / 3.0) * fsign(cosY);
    float mCosT = clamp(floor(gl_FragCoord.y - 1.0) / 256., 0.0, 1.0);
    float Y = facos(cosY);
    const float a = -0.8;
    const float b = -0.1;
    const float c = 3.0;
    const float d = -7.;
    const float e = 0.35;

  //luminance (cie model)
    vec3 daySky = vec3(0.0);
    vec3 moonSky = vec3(0.0);
	// Day
    if(skyIntensity > 0.00001) {
      float L0 = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * Y) - exp(d * 3.1415 / 2.)) + e * cosY * cosY);
      vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) / 1.5, vec3(0.4, 0.5, 0.6) / 1.5, rainStrength);
      vec3 normalizedSunColor = nsunColor;

      vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength * 0.5);
      daySky = pow(L0, 1.0 - rainStrength * 0.75) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * SKY_BRIGHTNESS_DAY;
    }
	// Night
    else if(skyIntensityNight > 0.00001) {
      float L0Moon = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * (PI - Y)) - exp(d * 3.1415 / 2.)) + e * cosY * cosY);
      moonSky = pow(L0Moon, 1.0 - rainStrength * 0.75) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) * SKY_BRIGHTNESS_NIGHT;
    }
    fragColor.rgb = (daySky + moonSky);

    fragColor.rgb = mix(fragColor.rgb * vec3(0.2, 0.2, 0.2) * 1.0, fragColor.rgb, 1 - rainStrength);

  }

}
