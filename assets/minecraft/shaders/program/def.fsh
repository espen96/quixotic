#version 150


in float skyIntensity;
in vec3 nsunColor;
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

float luminance(vec3 v) {
  return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 reinhard_jodie(vec3 v) {
  float l = luminance(v);
  vec3 tv = v / (1.0f + v);
  return mix(v / (1.0f + l), tv, tv);
}

float luma3(vec3 color) {
  return dot(color, vec3(0.21, 0.72, 0.07));
}
float decodeFloat7_4(uint raw) {
    uint sign = raw >> 11u;
    uint exponent = (raw >> 7u) & 15u;
    uint mantissa = 128u | (raw & 127u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 14.0);
}

float decodeFloat6_4(uint raw) {
    uint sign = raw >> 10u;
    uint exponent = (raw >> 6u) & 15u;
    uint mantissa = 64u | (raw & 63u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 13.0);
}

vec3 decodeColor(vec4 raw) {
    uvec4 scaled = uvec4(round(raw * 255.0));
    uint encoded = (scaled.r << 24) | (scaled.g << 16) | (scaled.b << 8) | scaled.a;

    return vec3(decodeFloat7_4(encoded >> 21), decodeFloat7_4((encoded >> 10) & 2047u), decodeFloat6_4(encoded & 1023u));
}

uint encodeFloat7_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 14.0)) & 127u;
    return (sign << 11u) | (exponent << 7u) | mantissa;
}

uint encodeFloat6_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 13.0)) & 63u;
    return (sign << 10u) | (exponent << 6u) | mantissa;
}

vec4 encodeColor(vec3 color) {
    uint r = encodeFloat7_4(color.r);
    uint g = encodeFloat7_4(color.g);
    uint b = encodeFloat6_4(color.b);

    uint encoded = (r << 21) | (g << 10) | b;
    return vec4(encoded >> 24, (encoded >> 16) & 255u, (encoded >> 8) & 255u, encoded & 255u) / 255.0;
}



float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}

void main() {

  //vec3 avgAmbient = (ambientUp + ambientLeft + ambientRight + ambientB + ambientF + ambientDown) / 6. * (1.0 + rainStrength * 0.2);
  vec4 outcol = vec4(0.0);
  if(gl_FragCoord.x < 17. && gl_FragCoord.y < 17.) {

    vec3 avgAmbient = ds + ms;
    avgAmbient = mix(avgAmbient * vec3(0.2, 0.2, 0.5) * 1.0, avgAmbient, 1 - rainStrength);
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

    outcol = vec4(reinhard_jodie(ambient * 10.0), 1.0);

  }
//Save light values
  if(gl_FragCoord.x < 1. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientUp, 1.0);
  if(gl_FragCoord.x > 1. && gl_FragCoord.x < 2. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientDown, 1.0);
  if(gl_FragCoord.x > 2. && gl_FragCoord.x < 3. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientLeft, 1.0);
  if(gl_FragCoord.x > 3. && gl_FragCoord.x < 4. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientRight, 1.0);
  if(gl_FragCoord.x > 4. && gl_FragCoord.x < 5. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientB, 1.0);
  if(gl_FragCoord.x > 5. && gl_FragCoord.x < 6. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(ambientF, 1.0);
  if(gl_FragCoord.x > 6. && gl_FragCoord.x < 7. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(lightSourceColor, 1.0);
  /*if(gl_FragCoord.x > 7. && gl_FragCoord.x < 8. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(avgAmbient, 1.0);*/
  if(gl_FragCoord.x > 8. && gl_FragCoord.x < 9. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1) 
    outcol = encodeColor(lightCol.rgb);
  

  if(gl_FragCoord.x > 11. && gl_FragCoord.x < 12. && gl_FragCoord.y > 19. + 18. && gl_FragCoord.y < 19. + 18. + 1)
    outcol = vec4(avgSky, 1.0);

  else if(gl_FragCoord.x > 18. && gl_FragCoord.y > 1.) {
    float cosY = clamp(floor(gl_FragCoord.x - 18.0) / 256. * 2.0 - 1.0, -0.99999, 0.99999);
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

      vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength);
      daySky = pow(L0, 1.0 - rainStrength) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * SKY_BRIGHTNESS_DAY;
    }
	// Night
    else if(skyIntensityNight > 0.00001) {
      float L0Moon = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * (PI - Y)) - exp(d * 3.1415 / 2.)) + e * cosY * cosY);
      moonSky = pow(L0Moon, 1.0 - rainStrength) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) * SKY_BRIGHTNESS_NIGHT;
    }
    outcol.rgb = (daySky + moonSky);

    //fragColor.rgb = mix(fragColor.rgb * vec3(0.2, 0.2, 0.2) * 1.0, fragColor.rgb, 1 - rainStrength);

  }
fragColor = outcol;
}
