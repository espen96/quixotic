#version 150

#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform vec4 ColorModulator;
uniform vec2 ScreenSize;
uniform mat4 ModelViewMat;

in mat4 gbufferModelViewInverse;
in vec3 cscale;
in vec4 test;
in vec3 c1;
in vec3 c2;

in vec3 c3;
in vec2 texCoord0;
in float isSun;

out vec4 fragColor;

#define PRECISIONSCALE 1000.0
#define MAGICSUNSIZE 3.0

void main() {
    gl_FragDepth = gl_FragCoord.z;
    vec4 color = vec4(0.0);
    bool gui = isGUI(ModelViewMat);
    int index = inControl(gl_FragCoord.xy, ScreenSize.x);

/*
    const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994)); //radians() is not a const function on some drivers, so multiply by pi/180 manually.

//minecraft's native calculateCelestialAngle() function, ported to GLSL.
    float ang = fract(worldTime / 24000.0 - 0.25);
    ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959; //0-2pi, rolls over from 2pi to 0 at noon.

    vec3 sunDirTemp = vec3(-sin(ang), cos(ang) * sunRotationData);
    sunDir = normalize(vec3(sunDirTemp.x, sunDir.y, sunDirTemp.z));

    float rainStrength = (1 - rain.r) * 0.5;
    vec3 sunDir2 = sunDir;
    sunPosition = sunDir2;

    sunPosition3 = mat3(ModeViewMat) * sunDir2;
    vec3 upPosition = vec3(0, 1, 0);
    sunVec = sunDir2;

    eyeBrightnessSmooth = vec2(240);

    float normSunVec = sqrt(sunPosition.x * sunPosition.x + sunPosition.y * sunPosition.y + sunPosition.z * sunPosition.z);
    float normUpVec = sqrt(upPosition.x * upPosition.x + upPosition.y * upPosition.y + upPosition.z * upPosition.z);

    float sunPosX = sunPosition.x / normSunVec;
    float sunPosY = sunPosition.y / normSunVec;
    float sunPosZ = sunPosition.z / normSunVec;

    float upPosX = upPosition.x / normUpVec;
    float upPosY = upPosition.y / normUpVec;
    float upPosZ = upPosition.z / normUpVec;

    sunElevation = sunPosX * upPosX + sunPosY * upPosY + sunPosZ * upPosZ;

    float modWT = (worldTime % 24000) * 1.0;
    float fogAmount0 = 1 / 3000. + FOG_TOD_MULTIPLIER * (1 / 180. * (clamp(modWT - 11000., 0., 2000.0) / 2000. + (1.0 - clamp(modWT, 0., 3000.0) / 3000.)) * (clamp(modWT - 11000., 0., 2000.0) / 2000. + (1.0 - clamp(modWT, 0., 3000.0) / 3000.)) + 1 / 200. * clamp(modWT - 13000., 0., 1000.0) / 1000. * (1.0 - clamp(modWT - 23000., 0., 1000.0) / 1000.));
    fogAmount = BASE_FOG_AMOUNT * (fogAmount0 + max(FOG_RAIN_MULTIPLIER * 1 / 20. * rainStrength, FOG_TOD_MULTIPLIER * 1 / 50. * clamp(modWT - 13000., 0., 1000.0) / 1000. * (1.0 - clamp(modWT - 23000., 0., 1000.0) / 1000.)));

    float angMoon = -((pi * 0.5128205128205128 - facos(-sunElevation * 1.065 - 0.065)) / 1.5);
    float angSun = -((pi * 0.5128205128205128 - facos(sunElevation * 1.065 - 0.065)) / 1.5);

    float sunElev = pow(clamp(1.0 - sunElevation, 0.0, 1.0), 4.0) * 1.8;
    const float sunlightR0 = 1.0;
    float sunlightG0 = (0.89 * exp(-sunElev * 0.57)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
    float sunlightB0 = (0.8 * exp(-sunElev * 1.4)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;

    float sunlightR = sunlightR0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightG = sunlightG0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightB = sunlightB0 / (sunlightR0 + sunlightG0 + sunlightB0);
    vec3 nsunColor = vec3(sunlightR, sunlightG, sunlightB);

///////////////////////////

    float angSkyNight = -((pi * 0.5128205128205128 - facos(-sunElevation * 0.95 + 0.05)) / 1.5);
    float angSky = -((pi * 0.5128205128205128 - facos(sunElevation * 0.95 + 0.05)) / 1.5);

    float fading = clamp(sunElevation + 0.095, 0.0, 0.08) / 0.08;
    float fading2 = clamp(-sunElevation + 0.095, 0.0, 0.08) / 0.08;
    float skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);
    float skyIntensityNight = max(0., 1.0 - exp(angSkyNight)) * (1.0 - rainStrength * 0.4) * pow(fading2, 5.0);

    float moonIntensity = max(0., 1.0 - exp(angMoon));
    float sunIntensity = max(0., 1.0 - exp(angSun));
    vec3 sunVec = vec3(sunPosX, sunPosY, sunPosZ);
    moonIntensity = max(0., 1.0 - exp(angMoon));

    nsunColor = vec3(sunlightR, sunlightG, sunlightB);
    float avgEyeIntensity = ((sunIntensity * 120. + moonIntensity * 4.) + skyIntensity * 230. + skyIntensityNight * 4.);
    float exposure = 0.18 / log(max(avgEyeIntensity * 0.16 + 1.0, 1.13)) * 0.3 * log(2.0);
    const float sunAmount = 27.0 * 1.5;
    float lightSign = clamp(sunIntensity * pow(10., 35.), 0., 1.);
    vec4 lightCol = vec4((sunlightR * 3. * sunAmount * sunIntensity + 0.16 / 5. - 0.16 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * 7.84 * exposure, 7.84 * (sunlightG * 3. * sunAmount * sunIntensity + 0.24 / 5. - 0.24 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, 7.84 * (sunlightB * 3. * sunAmount * sunIntensity + 0.36 / 5. - 0.36 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, lightSign * 2.0 - 1.0);

*/

    if(index != -1) {

        // store the sun position in eye space indices [0,2]
        if(isSun > 0.75 && index >= 0 && index <= 2) {

            vec4 sunDir = ModelViewMat * vec4(normalize(c1 / cscale.x + c3 / cscale.z), 0.0);

            color = vec4(encodeFloat(sunDir[index]), 1.0);

        } else if(isSun < 0.25) {
            color = texture(Sampler0, texCoord0) * ColorModulator;
        }

        if(isSun > 0.75 && index == 30) {
            color = vec4(ColorModulator.a, 0, 0, 1);

        }
        if(isSun > 0.75 && index == 31 && ColorModulator.a == 0.1) {
            color = vec4(1);

        }
        if(isSun > 0.75 && index == 99) {
            vec3 sunDir =  (ModelViewMat * vec4(normalize(c1 / cscale.x + c3 / cscale.z), 0.0)).xyz;

            bool time8 = sunDir.y > 0;
            float time4 = map(sunDir.x, -1, +1, 0, 1);
            float time5 = mix(12000, 0, time4);
            float time6 = mix(24000, 12000, 1 - time4);
            float time7 = mix(time6, time5, time8);

            int worldTime = int(time7);
            color = vec4(encodeFloat24(worldTime), 1);

        }
    }

    // calculate screen space UV of the sun since it was transformed to cover the entire screen in vsh so texCoord0 no longer works
    else if(isSun > 0.75) {

        discard;
        vec3 p1 = c1 / cscale.x;
        vec3 p2 = c2 / cscale.y;
        vec3 p3 = c3 / cscale.z;
        vec3 center = (p1 + p3) / (2 * PRECISIONSCALE); // scale down vector to reduce fp issues

        vec4 tmp = (gbufferModelViewInverse * vec4(2.0 * (gl_FragCoord.xy / ScreenSize - 0.5), 1.0, 1.0));
        vec3 planepos = tmp.xyz / tmp.w;
        float lookingat = dot(planepos, center);
        planepos = planepos / lookingat;
        vec2 uv = vec2(dot(p2 - p1, planepos - center), dot(p3 - p2, planepos - center));
        uv = uv / PRECISIONSCALE * MAGICSUNSIZE + vec2(0.5);

        // only draw one sun lol
        if(lookingat > 0.0 && all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)))) {
            color = texture(Sampler0, uv) * ColorModulator;
            color.a = 0.0;

        }
    } else {

        if(gl_FragCoord.z > 0.9)
            discard;
        color = texture(Sampler0, texCoord0) * ColorModulator;
    }

    if(color.a == 0.0) {
        discard;
    }

    fragColor = color;

}
