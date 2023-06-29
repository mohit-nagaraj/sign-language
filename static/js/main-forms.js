const button = document.querySelector(".upload");
const image = document.querySelector(".logo");

button.addEventListener("click", () => {
  const apiKey = "gewgneoiwjgoisjdiojgiosjiog";
  const client = filestack.init(apiKey);
  const options = {
    onUploadDone: (uploadResponse) => {
      console.log("onUploadDone", uploadResponse);
      image.setAttribute("src", uploadResponse.filesUploaded[0].url);
    },
  };
  client.picker(options).open();
});

import { FilestackModule } from '@filestack/angular';
@NgModule({
  ...
  imports: [

    FilestackModule
  ],
})
export class AppModule {}
